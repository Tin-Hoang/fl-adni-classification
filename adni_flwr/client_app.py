"""Client application for ADNI Federated Learning."""

import os
import time
import torch
import numpy as np
import json
from flwr.common import Context, FitRes, EvaluateRes, Status, Code
from flwr.client import NumPyClient, ClientApp
from typing import Dict, Tuple, List
from sklearn.metrics import confusion_matrix
from adni_classification.config.config import Config
from adni_classification.utils.torch_utils import set_seed
from adni_classification.utils.training_utils import get_scheduler
from adni_flwr.task import (
    load_model,
    load_data,
    train,
    test_with_predictions,
    get_params,
    set_params,
    create_criterion,
    safe_parameters_to_ndarrays
)
from adni_flwr.strategies import StrategyFactory, StrategyAwareClient


class ADNIClient(NumPyClient):
    """Federated Learning Client for ADNI classification."""

    def __init__(self, config: Config, device: torch.device):
        """Initialize the ADNI client.

        Args:
            config: Config object containing client configuration
            device: Device to use for computation
        """
        self.config = config
        self.device = device

        # Load the model
        self.model = load_model(self.config)

        # Set seed
        set_seed(self.config.training.seed)

        # Load the datasets and data loaders
        self.train_loader, self.val_loader = load_data(self.config, batch_size=self.config.training.batch_size)

        # Determine the number of local epochs for this client
        self.local_epochs = self.config.fl.local_epochs

        # Get evaluation frequency (default to 1 if not specified)
        # Client ID must be explicitly set - FAIL FAST if not specified
        if not hasattr(self.config.fl, 'client_id') or self.config.fl.client_id is None:
            raise ValueError(
                "ERROR: 'client_id' not specified in client config. "
                "You must explicitly set 'client_id' in the FL config section. "
                "This prevents client identification issues in federated learning."
            )
        self.client_id = self.config.fl.client_id

        self.evaluate_frequency = getattr(self.config.fl, 'evaluate_frequency', 1)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Get the train dataset from the loader to create the criterion
        self.train_dataset = self.train_loader.dataset

        # Create criterion (loss function)
        self.criterion = create_criterion(
            self.config,
            self.train_dataset,
            self.device
        )

        # Initialize learning rate scheduler
        scheduler_type = getattr(self.config.training, 'lr_scheduler', None)
        self.scheduler = get_scheduler(
            scheduler_type=scheduler_type if scheduler_type else "none",
            optimizer=self.optimizer,
            num_epochs=self.local_epochs
        )

        # Get other training parameters
        self.mixed_precision = self.config.training.mixed_precision
        self.gradient_accumulation_steps = self.config.training.gradient_accumulation_steps

        # Initialize checkpoint functionality
        self.checkpoint_dir = config.checkpoint_dir
        self.client_checkpoint_dir = os.path.join(self.checkpoint_dir, f"client_{self.client_id}")
        os.makedirs(self.client_checkpoint_dir, exist_ok=True)

        # Checkpoint saving configuration
        self.save_client_checkpoints = getattr(self.config.training.checkpoint, 'save_regular', True)
        self.checkpoint_save_frequency = getattr(self.config.training.checkpoint, 'save_frequency', 10)

        # Track training state
        self.current_round = 0
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'rounds': []
        }

        print(f"Initialized ADNI client with config: {self.config.wandb.run_name if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'run_name') else 'unknown'}")
        print(f"Train dataset size: {len(self.train_loader.dataset)}")
        print(f"Validation dataset size: {len(self.val_loader.dataset)}")
        print(f"Using scheduler: {scheduler_type if scheduler_type else 'none'}")
        print(f"Evaluation frequency: every {self.evaluate_frequency} round(s)")
        print(f"Client checkpoint directory: {self.client_checkpoint_dir}")
        print(f"Client checkpoint saving: {'enabled' if self.save_client_checkpoints else 'disabled'} (frequency: {self.checkpoint_save_frequency})")

    def _save_client_checkpoint(self, round_num: int, train_loss: float, train_acc: float, is_best: bool = False):
        """Save client checkpoint after local training.

        Args:
            round_num: Current FL round number
            train_loss: Training loss from this round
            train_acc: Training accuracy from this round
            is_best: Whether this is the best model so far
        """
        if not self.save_client_checkpoints:
            return

        try:
            checkpoint = {
                'round': round_num,
                'client_id': self.client_id,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'best_val_accuracy': self.best_val_accuracy,
                'training_history': self.training_history,
                'config': {
                    'local_epochs': self.local_epochs,
                    'learning_rate': self.config.training.learning_rate,
                    'weight_decay': self.config.training.weight_decay,
                }
            }

            # Add scheduler state if available
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            # Save regular checkpoint based on frequency
            if round_num % self.checkpoint_save_frequency == 0:
                checkpoint_path = os.path.join(self.client_checkpoint_dir, f"checkpoint_round_{round_num}.pt")
                torch.save(checkpoint, checkpoint_path)
                print(f"Client {self.client_id}: Saved checkpoint for round {round_num} to {checkpoint_path}")

            # Always save latest checkpoint (overwrite)
            latest_path = os.path.join(self.client_checkpoint_dir, "checkpoint_latest.pt")
            torch.save(checkpoint, latest_path)

            # Save best checkpoint if this is the best model
            if is_best:
                best_path = os.path.join(self.client_checkpoint_dir, "checkpoint_best.pt")
                torch.save(checkpoint, best_path)
                print(f"Client {self.client_id}: Saved new best checkpoint with accuracy {train_acc:.2f}% to {best_path}")

        except Exception as e:
            print(f"Client {self.client_id}: Error saving checkpoint: {e}")

    def _load_client_checkpoint(self, checkpoint_path: str) -> bool:
        """Load client checkpoint to resume training.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                print(f"Client {self.client_id}: Checkpoint file not found: {checkpoint_path}")
                return False

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Restore training state
            self.current_round = checkpoint.get('round', 0)
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            self.training_history = checkpoint.get('training_history', {
                'train_losses': [],
                'train_accuracies': [],
                'val_losses': [],
                'val_accuracies': [],
                'rounds': []
            })

            print(f"Client {self.client_id}: Successfully loaded checkpoint from round {self.current_round}")
            print(f"Client {self.client_id}: Best validation accuracy so far: {self.best_val_accuracy:.2f}%")
            return True

        except Exception as e:
            print(f"Client {self.client_id}: Error loading checkpoint: {e}")
            return False

    def resume_from_checkpoint(self) -> bool:
        """Attempt to resume training from the latest checkpoint.

        Returns:
            True if resumed successfully, False if no checkpoint found or loading failed
        """
        # If model.pretrained_checkpoint is specified and is an FL checkpoint, use it
        if self.config.model.pretrained_checkpoint:
            from adni_flwr.task import is_fl_client_checkpoint
            if is_fl_client_checkpoint(self.config.model.pretrained_checkpoint):
                print(f"Resuming from specified FL checkpoint: {self.config.model.pretrained_checkpoint}")
                return self._load_client_checkpoint(self.config.model.pretrained_checkpoint)

        # Otherwise, try the latest checkpoint
        latest_checkpoint = os.path.join(self.client_checkpoint_dir, "checkpoint_latest.pt")
        return self._load_client_checkpoint(latest_checkpoint)

    def fit(self, parameters, config) -> FitRes:
        """Train the model on the local dataset.

        Args:
            parameters: Model parameters from the server
            config: Configuration from the server for this round

        Returns:
            Updated model parameters, number of training examples, metrics
        """
        # Get current round number from config
        current_round = config.get("server_round", self.current_round + 1)
        self.current_round = current_round

        # Update local model with global parameters
        # Convert parameters to numpy arrays safely
        param_arrays = safe_parameters_to_ndarrays(parameters)

        set_params(self.model, param_arrays)

        # Override local epochs if specified in the config
        local_epochs = config.get("local_epochs", self.local_epochs)

        # Train the model
        start_time = time.time()
        train_loss, train_acc, current_lr = train(
            model=self.model,
            train_loader=self.train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            epoch_num=local_epochs,
            mixed_precision=self.config.training.mixed_precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            scheduler=self.scheduler
        )
        end_time = time.time()
        training_time = end_time - start_time

        # Update training history
        self.training_history['train_losses'].append(train_loss)
        self.training_history['train_accuracies'].append(train_acc)
        self.training_history['rounds'].append(current_round)

        # Check if this is the best training accuracy (simple heuristic)
        is_best = train_acc > self.best_val_accuracy
        if is_best:
            self.best_val_accuracy = train_acc

        # Save client checkpoint
        self._save_client_checkpoint(current_round, train_loss, train_acc, is_best)

        # Log training metrics
        print(f"Client {self.client_id} training round {current_round}: loss={train_loss:.4f}, accuracy={train_acc:.2f}%, training time={training_time:.2f} seconds")

        # Return updated model parameters and metrics
        return get_params(self.model), len(self.train_loader.dataset), {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_lr": float(current_lr),
            "client_id": self.client_id,
            "round": current_round,
            "training_time": training_time,
        }

    def test_serialization(self, metrics: Dict) -> bool:
        """Test if the metrics can be serialized to JSON.

        Args:
            metrics: Dictionary of metrics to test

        Returns:
            True if serialization is successful, False otherwise
        """
        try:
            json_str = json.dumps(metrics)
            # Reconstruct to verify
            json.loads(json_str)
            print(f"Serialization test passed. Size: {len(json_str)} bytes")
            return True
        except Exception as e:
            print(f"ERROR: Serialization test failed: {e}")

            # Try to identify problematic keys
            for k, v in metrics.items():
                try:
                    json.dumps({k: v})
                except Exception as sub_e:
                    print(f"  Problem with key '{k}': {sub_e}")
                    print(f"  Type: {type(v)}, Value preview: {str(v)[:100]}")

            return False

    def evaluate(self, parameters, config) -> EvaluateRes:
        """Evaluate the model on the local validation dataset.

        Args:
            parameters: Model parameters from the server
            config: Configuration from the server for this round

        Returns:
            Loss, number of evaluation examples, metrics
        """
        # Get the current round number from the config
        current_round = config.get("server_round", 1)

        # Check if we should evaluate in this round
        if current_round % self.evaluate_frequency != 0:
            print(f"Client {self.client_id}: Skipping evaluation for round {current_round} (evaluating every {self.evaluate_frequency} rounds)")
            # Return a minimal result indicating no evaluation was performed
            return 0.0, 0, {
                "client_id": str(self.client_id),
                "evaluation_skipped": True,
                "evaluation_frequency": self.evaluate_frequency,
                "current_round": current_round
            }

        print(f"Client {self.client_id}: Performing evaluation for round {current_round}")

        try:
            # Update local model with global parameters
            # Convert parameters to numpy arrays safely
            param_arrays = safe_parameters_to_ndarrays(parameters)

            set_params(self.model, param_arrays)

            # Evaluate the model to get predictions and true labels
            val_loss, val_acc, predictions, true_labels = test_with_predictions(
                model=self.model,
                test_loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                mixed_precision=self.config.training.mixed_precision
            )

            # Update validation history
            self.training_history['val_losses'].append(val_loss)
            self.training_history['val_accuracies'].append(val_acc)

            # Log evaluation metrics
            print(f"Client {self.client_id} evaluation round {current_round}: loss={val_loss:.4f}, accuracy={val_acc:.2f}%")

            # Print information about predictions and labels for debugging
            print(f"Client {self.client_id}: Predictions length={len(predictions)}, Labels length={len(true_labels)}")
            print(f"Client {self.client_id}: First 5 predictions={predictions[:5]}, First 5 labels={true_labels[:5]}")

            # Convert to Python native types (especially important for numpy types)
            predictions_list = [int(p) for p in predictions]
            labels_list = [int(l) for l in true_labels]

            # Determine if we need to sample to reduce message size
            max_samples = 500  # Limit to stay within message size constraints
            if len(predictions_list) > max_samples:
                # Random sample for large datasets
                import random
                indices = sorted(random.sample(range(len(predictions_list)), max_samples))
                predictions_sample = [predictions_list[i] for i in indices]
                labels_sample = [labels_list[i] for i in indices]
                sample_info = f"sampled_{max_samples}_from_{len(predictions_list)}"
            else:
                predictions_sample = predictions_list
                labels_sample = labels_list
                sample_info = "full_dataset"

            # Serialize to JSON strings
            predictions_json = json.dumps(predictions_sample)
            labels_json = json.dumps(labels_sample)

            print(f"Client {self.client_id}: Serialized predictions length={len(predictions_json)} bytes")
            print(f"Client {self.client_id}: Serialized labels length={len(labels_json)} bytes")

            # Calculate confusion matrix locally for backup/debugging
            try:
                cm = confusion_matrix(true_labels, predictions)
                print(f"Client {self.client_id}: Local confusion matrix:\n{cm}")

                # You can still save it to a file as backup
                os.makedirs("client_matrices", exist_ok=True)
                np.save(f"client_matrices/confusion_matrix_client_{self.client_id}.npy", cm)
            except Exception as e:
                print(f"Client {self.client_id}: Error creating local confusion matrix: {e}")

            # Create result dictionary with encoded data
            result = {
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "predictions_json": predictions_json,
                "labels_json": labels_json,
                "sample_info": sample_info,
                "client_id": str(self.client_id),
                "num_classes": 2 if self.config.data.classification_mode == "CN_AD" else 3,
                "evaluation_skipped": False,
                "evaluation_frequency": self.evaluate_frequency,
                "current_round": current_round
            }

            # Test serialization for safety
            success = self.test_serialization(result)
            if not success:
                # Fall back to minimal metrics if serialization fails
                print(f"Client {self.client_id}: WARNING - Serialization failed, falling back to minimal metrics")
                result = {
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "client_id": str(self.client_id),
                    "error": "Serialization failed",
                    "evaluation_skipped": False,
                    "evaluation_frequency": self.evaluate_frequency,
                    "current_round": current_round
                }

            return float(val_loss), len(self.val_loader.dataset), result

        except Exception as e:
            import traceback
            print(f"Client {self.client_id}: Error in evaluate method: {e}")
            print(traceback.format_exc())
            # Return minimal results to avoid failure
            return 0.0, 0, {
                "client_id": str(self.client_id),
                "error": str(e),
                "evaluation_skipped": False,
                "evaluation_frequency": self.evaluate_frequency,
                "current_round": current_round
            }


def client_fn(context: Context):
    """Client factory function.

    Args:
        context: Context containing client configuration

    Returns:
        An instance of NumPyClient
    """
    # Print the context
    print(f"Context: {context}")

    # Determine which GPU to use if available
    gpu_idx = context.node_config.get("gpu-id", 0)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    # Get partition ID to determine which config file to use
    partition_id = context.node_config.get("partition-id", 0)

    # Get client config files from app config
    client_config_files = context.run_config.get("client-config-files", "")
    if isinstance(client_config_files, str):
        client_config_files = [s.strip() for s in client_config_files.split(",") if s.strip()]

    # Ensure we have enough config files for all partitions
    if partition_id >= len(client_config_files):
        raise ValueError(f"Partition ID {partition_id} is out of range for {len(client_config_files)} client config files")

    # Get the specific config file for this client
    config_path = client_config_files[partition_id]
    config = Config.from_yaml(config_path)

    # Determine which strategy to use - FAIL FAST if not specified
    if not hasattr(config.fl, 'strategy') or not config.fl.strategy:
        raise ValueError(
            f"ERROR: 'strategy' not specified in client config {config_path}. "
            f"You must explicitly set 'strategy' in the FL config section. "
            f"Available strategies: fedavg, fedprox, secagg. "
            f"This prevents dangerous implicit defaults that could cause strategy mismatch between clients and server."
        )

    strategy_name = config.fl.strategy
    print(f'Initializing client {partition_id} with {strategy_name} strategy, config: {config_path} on device: {device}')

    # Check if we should use the new strategy system - FAIL FAST if not specified
    if not hasattr(config.fl, 'use_strategy_system'):
        raise ValueError(
            f"ERROR: 'use_strategy_system' not specified in client config {config_path}. "
            f"You must explicitly set 'use_strategy_system: true' or 'use_strategy_system: false' "
            f"in the FL config section to choose between new strategy system or legacy client."
        )

    if config.fl.use_strategy_system:
        # New strategy system path
        print(f"Using new strategy system with {strategy_name} strategy")

        # Load model and create optimizer/criterion
        model = load_model(config)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Load data to create criterion
        train_loader, _ = load_data(config, batch_size=config.training.batch_size)
        criterion = create_criterion(config, train_loader.dataset, device)

        # Create client strategy
        client_strategy = StrategyFactory.create_client_strategy(
            strategy_name=strategy_name,
            config=config,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        # Create strategy-aware client
        client = StrategyAwareClient(config=config, device=device, client_strategy=client_strategy)

        # Attempt to resume from FL client checkpoint if specified
        if config.model.pretrained_checkpoint:
            from adni_flwr.task import is_fl_client_checkpoint
            if is_fl_client_checkpoint(config.model.pretrained_checkpoint):
                print(f"Attempting to resume strategy-aware client {partition_id} from FL checkpoint: {config.model.pretrained_checkpoint}")
                if client.resume_from_checkpoint():
                    print(f"Strategy-aware client {partition_id} successfully resumed from FL checkpoint")
                else:
                    print(f"Strategy-aware client {partition_id} starting fresh (FL checkpoint loading failed)")
            else:
                print(f"Strategy-aware client {partition_id}: pretrained_checkpoint is a regular model checkpoint, not resuming FL state")

        return client.to_client()
    else:
        # Legacy path: use original ADNIClient for backward compatibility
        print(f"Using legacy ADNIClient for backward compatibility")
        client = ADNIClient(config=config, device=device)

        # Attempt to resume from FL client checkpoint if specified
        if config.model.pretrained_checkpoint:
            from adni_flwr.task import is_fl_client_checkpoint
            if is_fl_client_checkpoint(config.model.pretrained_checkpoint):
                print(f"Attempting to resume client {partition_id} from FL checkpoint: {config.model.pretrained_checkpoint}")
                if client.resume_from_checkpoint():
                    print(f"Client {partition_id} successfully resumed from FL checkpoint")
                else:
                    print(f"Client {partition_id} starting fresh (FL checkpoint loading failed)")
            else:
                print(f"Client {partition_id}: pretrained_checkpoint is a regular model checkpoint, not resuming FL state")

        return client.to_client()


# Initialize the client app
app = ClientApp(client_fn=client_fn)
