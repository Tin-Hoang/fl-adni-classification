"""Client application for ADNI Federated Learning."""

import os
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
    create_criterion
)


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
        self.client_id = self.config.fl.client_id if hasattr(self.config.fl, 'client_id') else 'unknown'

        # Load the model
        self.model = load_model(self.config)

        # Set seed
        set_seed(self.config.training.seed)

        # Load the datasets and data loaders
        self.train_loader, self.val_loader = load_data(self.config, batch_size=self.config.training.batch_size)

        # Determine the number of local epochs for this client
        self.local_epochs = self.config.fl.local_epochs

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

        print(f"Initialized ADNI client with config: {self.config.wandb.run_name if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'run_name') else 'unknown'}")
        print(f"Train dataset size: {len(self.train_loader.dataset)}")
        print(f"Validation dataset size: {len(self.val_loader.dataset)}")
        print(f"Using scheduler: {scheduler_type if scheduler_type else 'none'}")

    def fit(self, parameters, config) -> FitRes:
        """Train the model on the local dataset.

        Args:
            parameters: Model parameters from the server
            config: Configuration from the server for this round

        Returns:
            Updated model parameters, number of training examples, metrics
        """
        # Update local model with global parameters
        set_params(self.model, parameters)

        # Override local epochs if specified in the config
        local_epochs = config.get("local_epochs", self.local_epochs)

        # Train the model
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

        # Log training metrics
        print(f"Client training: loss={train_loss:.4f}, accuracy={train_acc:.2f}%")

        # Return updated model parameters and metrics
        return get_params(self.model), len(self.train_loader.dataset), {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_lr": float(current_lr),
            "client_id": self.client_id,
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
        try:
            # Update local model with global parameters
            set_params(self.model, parameters)

            # Evaluate the model to get predictions and true labels
            val_loss, val_acc, predictions, true_labels = test_with_predictions(
                model=self.model,
                test_loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                mixed_precision=self.config.training.mixed_precision
            )

            # Log evaluation metrics
            print(f"Client evaluation: loss={val_loss:.4f}, accuracy={val_acc:.2f}%")

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
                "num_classes": 2 if self.config.data.classification_mode == "CN_AD" else 3
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
                    "error": "Serialization failed"
                }

            return float(val_loss), len(self.val_loader.dataset), result

        except Exception as e:
            import traceback
            print(f"Client {self.client_id}: Error in evaluate method: {e}")
            print(traceback.format_exc())
            # Return minimal results to avoid failure
            return 0.0, 0, {"client_id": str(self.client_id), "error": str(e)}


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
    print(f'Initializing client {partition_id} with config: {config_path} on device: {device}')
    client = ADNIClient(config=Config.from_yaml(config_path), device=device)
    return client.to_client()


# Initialize the client app
app = ClientApp(client_fn=client_fn)
