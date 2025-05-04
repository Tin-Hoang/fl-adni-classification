"""Client application for ADNI Federated Learning."""

import os
import torch
from flwr.common import Context
from flwr.client import NumPyClient, ClientApp

from adni_flwr.task import (
    load_config_from_yaml,
    load_model,
    load_data,
    train,
    test,
    get_params,
    set_params,
    create_criterion
)


class ADNIClient(NumPyClient):
    """Federated Learning Client for ADNI classification."""

    def __init__(self, config_path, device):
        """Initialize the ADNI client.

        Args:
            config_path: Path to the client configuration file
            device: Device to use for computation
        """
        self.config = load_config_from_yaml(config_path)
        self.device = device
        self.client_id = self.config.get("fl", {}).get("client_id", "unknown")

        # Load the model
        self.model = load_model(self.config)

        # Load the datasets and data loaders
        self.train_loader, self.val_loader = load_data(self.config)

        # Determine the number of local epochs for this client
        self.local_epochs = self.config.get("fl", {}).get("local_epochs", 1)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"].get("weight_decay", 0.0001),
        )

        # Get the train dataset from the loader to create the criterion
        self.train_dataset = self.train_loader.dataset

        # Create criterion (loss function)
        self.criterion = create_criterion(
            self.config,
            self.train_dataset,
            self.device
        )

        # Get other training parameters
        self.mixed_precision = self.config["training"].get("mixed_precision", False)
        self.gradient_accumulation_steps = self.config["training"].get("gradient_accumulation_steps", 1)

        print(f"Initialized ADNI client with config: {config_path}")
        print(f"Train dataset size: {len(self.train_loader.dataset)}")
        print(f"Validation dataset size: {len(self.val_loader.dataset)}")

    def fit(self, parameters, config):
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
        train_loss, train_acc = train(
            model=self.model,
            train_loader=self.train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            epoch_num=local_epochs,
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )

        # Log training metrics
        print(f"Client training: loss={train_loss:.4f}, accuracy={train_acc:.2f}%")

        # Return updated model parameters and metrics
        return get_params(self.model), len(self.train_loader.dataset), {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "client_id": self.client_id,
        }

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation dataset.

        Args:
            parameters: Model parameters from the server
            config: Configuration from the server for this round

        Returns:
            Loss, number of evaluation examples, metrics
        """
        # Update local model with global parameters
        set_params(self.model, parameters)

        # Evaluate the model
        val_loss, val_acc = test(
            model=self.model,
            test_loader=self.val_loader,
            criterion=self.criterion,
            device=self.device,
            mixed_precision=self.mixed_precision
        )

        # Log evaluation metrics
        print(f"Client evaluation: loss={val_loss:.4f}, accuracy={val_acc:.2f}%")

        # Return evaluation metrics
        return float(val_loss), len(self.val_loader.dataset), {
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "client_id": self.client_id,
        }


def client_fn(context: Context):
    """Client factory function.

    Args:
        context: Context containing client configuration

    Returns:
        An instance of NumPyClient
    """
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

    print(f"Initializing client {partition_id} with config: {config_path} on device: {device}")

    # Create and return the client
    client = ADNIClient(config_path=config_path, device=device)
    return client.to_client()


# Initialize the client app
app = ClientApp(client_fn=client_fn)
