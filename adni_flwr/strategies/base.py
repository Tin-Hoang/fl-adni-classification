"""Base classes for Federated Learning strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.common import Parameters, FitRes, EvaluateRes
from flwr.server.strategy import Strategy
from flwr.client import NumPyClient
import os
import time
import numpy as np

from adni_classification.config.config import Config
from adni_classification.utils.torch_utils import set_seed
from adni_classification.utils.training_utils import get_scheduler
from adni_flwr.task import (
    load_model,
    load_data,
    create_criterion,
    safe_parameters_to_ndarrays,
    debug_model_architecture,
    set_params
)


class FLStrategyBase(Strategy, ABC):
    """Base class for server-side FL strategies."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        wandb_logger: Optional[Any] = None,
        **kwargs
    ):
        """Initialize the FL strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            wandb_logger: Wandb logger instance
            **kwargs: Additional strategy-specific parameters
        """
        self.config = config
        self.model = model
        self.wandb_logger = wandb_logger
        self.strategy_config = kwargs

        super().__init__()

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        pass

    @abstractmethod
    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy-specific parameters."""
        pass

    def log_strategy_metrics(self, metrics: Dict[str, Any], round_num: int):
        """Log strategy-specific metrics.

        Args:
            metrics: Metrics to log
            round_num: Current round number
        """
        if self.wandb_logger:
            strategy_metrics = {
                f"strategy/{k}": v for k, v in metrics.items()
            }
            self.wandb_logger.log_metrics(strategy_metrics, step=round_num)


class ClientStrategyBase(ABC):
    """Base class for client-side FL strategies."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs
    ):
        """Initialize the client strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to use for computation
            scheduler: Learning rate scheduler (optional)
            **kwargs: Additional strategy-specific parameters
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.strategy_config = kwargs

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        pass

    @abstractmethod
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        **kwargs
    ) -> Tuple[float, float]:
        """Train the model for one epoch using the strategy.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            total_epochs: Total number of epochs
            **kwargs: Additional training parameters

        Returns:
            Tuple of (loss, accuracy)
        """
        pass

    @abstractmethod
    def prepare_for_round(self, server_params: Parameters, round_config: Dict[str, Any]):
        """Prepare the client for a new training round.

        Args:
            server_params: Parameters from server
            round_config: Configuration for this round
        """
        pass

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Return strategy-specific metrics.

        Returns:
            Dictionary of strategy metrics
        """
        return {
            "strategy_name": self.get_strategy_name(),
            **self.get_custom_metrics()
        }

    @abstractmethod
    def get_custom_metrics(self) -> Dict[str, Any]:
        """Return custom strategy-specific metrics.

        Returns:
            Dictionary of custom metrics
        """
        pass

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return strategy-specific data to be saved in checkpoints.

        This method can be overridden by specific strategies to save
        additional state information (e.g., global model parameters for FedProx).

        Returns:
            Dictionary of strategy-specific checkpoint data
        """
        return {}

    def load_checkpoint_data(self, checkpoint_data: Dict[str, Any]):
        """Load strategy-specific data from checkpoints.

        This method can be overridden by specific strategies to restore
        additional state information.

        Args:
            checkpoint_data: Strategy-specific checkpoint data
        """
        pass


class StrategyAwareClient(NumPyClient):
    """Base client class that uses strategy pattern for training."""

    def __init__(
        self,
        config: Config,
        device: torch.device,
        client_strategy: ClientStrategyBase
    ):
        """Initialize strategy-aware client.

        Args:
            config: Configuration object
            device: Device to use for computation
            client_strategy: Client-side strategy implementation
        """
        self.config = config
        self.device = device
        self.client_strategy = client_strategy
        # Client ID must be explicitly set - FAIL FAST if not specified
        if not hasattr(config.fl, 'client_id') or config.fl.client_id is None:
            raise ValueError(
                "ERROR: 'client_id' not specified in client config. "
                "You must explicitly set 'client_id' in the FL config section. "
                "This prevents client identification issues in federated learning."
            )
        self.client_id = config.fl.client_id

        # Load data
        from adni_flwr.task import load_data
        self.train_loader, self.val_loader = load_data(
            config, batch_size=config.training.batch_size
        )

        # Get evaluation frequency (default to 1 if not specified)
        self.evaluate_frequency = getattr(self.config.fl, 'evaluate_frequency', 1)

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

        print(f"Initialized {self.client_strategy.get_strategy_name()} client with config: {self.config.wandb.run_name if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'run_name') else 'unknown'}")
        print(f"Train dataset size: {len(self.train_loader.dataset)}")
        print(f"Validation dataset size: {len(self.val_loader.dataset)}")
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
                'model_state_dict': self.client_strategy.model.state_dict(),
                'optimizer_state_dict': self.client_strategy.optimizer.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'best_val_accuracy': self.best_val_accuracy,
                'training_history': self.training_history,
                'strategy_name': self.client_strategy.get_strategy_name(),
                'strategy_metrics': self.client_strategy.get_custom_metrics(),
                'config': {
                    'local_epochs': self.config.fl.local_epochs,
                    'learning_rate': self.config.training.learning_rate,
                    'weight_decay': self.config.training.weight_decay,
                }
            }

            # Add scheduler state if available
            if hasattr(self.client_strategy, 'scheduler') and self.client_strategy.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.client_strategy.scheduler.state_dict()

            # Add strategy-specific checkpoint data
            if hasattr(self.client_strategy, 'get_checkpoint_data'):
                checkpoint['strategy_data'] = self.client_strategy.get_checkpoint_data()

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

            # Verify strategy compatibility
            saved_strategy = checkpoint.get('strategy_name', 'unknown')
            current_strategy = self.client_strategy.get_strategy_name()
            if saved_strategy != current_strategy:
                print(f"Client {self.client_id}: Strategy mismatch - saved: {saved_strategy}, current: {current_strategy}")
                return False

            # Load model state
            self.client_strategy.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.client_strategy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state if available
            if ('scheduler_state_dict' in checkpoint and
                hasattr(self.client_strategy, 'scheduler') and
                self.client_strategy.scheduler is not None):
                self.client_strategy.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Client {self.client_id}: Restored scheduler state")

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

            # Load strategy-specific checkpoint data
            if hasattr(self.client_strategy, 'load_checkpoint_data') and 'strategy_data' in checkpoint:
                self.client_strategy.load_checkpoint_data(checkpoint['strategy_data'])

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
        """Train the model using the strategy.

        Args:
            parameters: Model parameters from server
            config: Round configuration

        Returns:
            Updated parameters and metrics
        """
        # Get current round number from config
        current_round = config.get("server_round", self.current_round + 1)
        self.current_round = current_round

        # Prepare for training round
        self.client_strategy.prepare_for_round(parameters, config)

        # Get local epochs
        local_epochs = config.get("local_epochs", self.config.fl.local_epochs)

        # Train using strategy
        total_loss = 0.0
        total_acc = 0.0

        start_time = time.time()
        for epoch in range(local_epochs):
            loss, acc = self.client_strategy.train_epoch(
                self.train_loader, epoch, local_epochs
            )
            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / local_epochs
        avg_acc = total_acc / local_epochs

        # Update training history
        self.training_history['train_losses'].append(avg_loss)
        self.training_history['train_accuracies'].append(avg_acc)
        self.training_history['rounds'].append(current_round)

        # Check if this is the best training accuracy (simple heuristic)
        is_best = avg_acc > self.best_val_accuracy
        if is_best:
            self.best_val_accuracy = avg_acc

        # Save client checkpoint
        self._save_client_checkpoint(current_round, avg_loss, avg_acc, is_best)

        # Get updated parameters (strategy-specific for SecAgg)
        if hasattr(self.client_strategy, 'get_secure_parameters'):
            # For SecAgg, get masked parameters
            updated_params = self.client_strategy.get_secure_parameters()
        else:
            # For other strategies, get regular parameters
            from adni_flwr.task import get_params
            updated_params = get_params(self.client_strategy.model)

        end_time = time.time()
        training_time = end_time - start_time

        # Log training metrics
        print(f"Client {self.client_id} training round {current_round}: loss={avg_loss:.4f}, accuracy={avg_acc:.2f}%, training_time={training_time:.2f} seconds")

        # Get current learning rate (if available)
        current_lr = self.client_strategy.optimizer.param_groups[0]['lr'] if hasattr(self.client_strategy, 'optimizer') else 0.0

        # Collect metrics
        metrics = {
            "train_loss": float(avg_loss),
            "train_accuracy": float(avg_acc),
            "train_lr": float(current_lr),
            "client_id": self.client_id,
            "round": current_round,
            "training_time": float(training_time),
            **self.client_strategy.get_strategy_metrics()
        }

        return updated_params, len(self.train_loader.dataset), metrics

    def test_serialization(self, metrics: Dict) -> bool:
        """Test if the metrics can be serialized to JSON.

        Args:
            metrics: Dictionary of metrics to test

        Returns:
            True if serialization is successful, False otherwise
        """
        try:
            import json
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
                    import json
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
        import json
        import os
        import numpy as np
        from sklearn.metrics import confusion_matrix

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
                "current_round": current_round,
                **self.client_strategy.get_strategy_metrics()
            }

        print(f"Client {self.client_id}: Performing evaluation for round {current_round}")

        try:
            # Update model with server parameters
            from adni_flwr.task import set_params, test_with_predictions, safe_parameters_to_ndarrays

            # Convert parameters to numpy arrays safely
            param_arrays = safe_parameters_to_ndarrays(parameters)

            set_params(self.client_strategy.model, param_arrays)

            # Evaluate the model to get predictions and true labels
            val_loss, val_acc, predictions, true_labels = test_with_predictions(
                model=self.client_strategy.model,
                test_loader=self.val_loader,
                criterion=self.client_strategy.criterion,
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
                "current_round": current_round,
                **self.client_strategy.get_strategy_metrics()
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
                    "current_round": current_round,
                    **self.client_strategy.get_strategy_metrics()
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
                "current_round": current_round,
                **self.client_strategy.get_strategy_metrics()
            }
