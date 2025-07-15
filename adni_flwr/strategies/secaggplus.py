"""Real Secure Aggregation (SecAgg+) strategy implementation using Flower's built-in SecAgg."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.common import Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters, FitIns, EvaluateIns, Context
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.workflow import SecAggPlusWorkflow
from flwr.client import NumPyClient
from flwr.client.mod import secaggplus_mod

from .base import FLStrategyBase, ClientStrategyBase
from adni_classification.config.config import Config
from adni_flwr.task import set_params, get_params, safe_parameters_to_ndarrays, load_data, test_with_predictions, create_criterion
from adni_classification.utils.visualization import plot_confusion_matrix

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SecAggPlusStrategy(FLStrategyBase):
    """Server-side Real Secure Aggregation (SecAgg+) strategy using Flower's built-in SecAgg."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        wandb_logger: Optional[Any] = None,
        num_shares: int = 3,
        reconstruction_threshold: int = 3,
        max_weight: int = 16777216,  # 2^24
        timeout: Optional[float] = None,
        clipping_range: float = 1.0,
        quantization_range: int = 2**20,
        **kwargs
    ):
        """Initialize SecAgg+ strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            wandb_logger: Wandb logger instance
            num_shares: Number of shares for secret sharing (must be >= reconstruction_threshold)
            reconstruction_threshold: Minimum number of shares needed for reconstruction
            max_weight: Maximum weight value for quantization
            timeout: Timeout for SecAgg operations
            clipping_range: Range for gradient clipping
            quantization_range: Range for quantization
            **kwargs: Additional SecAgg parameters
        """
        super().__init__(config, model, wandb_logger, **kwargs)

        # SecAgg+ specific parameters
        self.num_shares = num_shares
        self.reconstruction_threshold = reconstruction_threshold
        self.max_weight = max_weight
        self.timeout = timeout
        self.clipping_range = clipping_range
        self.quantization_range = quantization_range

        # Validate SecAgg parameters
        if self.num_shares < self.reconstruction_threshold:
            raise ValueError(f"num_shares ({num_shares}) must be >= reconstruction_threshold ({reconstruction_threshold})")

        if self.reconstruction_threshold < 1:
            raise ValueError(f"reconstruction_threshold ({reconstruction_threshold}) must be >= 1")

        # Extract specific parameters for FedAvg
        fedavg_params = {
            'fraction_fit': getattr(config.fl, 'fraction_fit', 1.0),
            'fraction_evaluate': getattr(config.fl, 'fraction_evaluate', 1.0),
            'min_fit_clients': getattr(config.fl, 'min_fit_clients', 2),
            'min_evaluate_clients': getattr(config.fl, 'min_evaluate_clients', 2),
            'min_available_clients': getattr(config.fl, 'min_available_clients', 2),
        }

        # Add aggregation functions if provided
        if 'evaluate_metrics_aggregation_fn' in kwargs:
            fedavg_params['evaluate_metrics_aggregation_fn'] = kwargs['evaluate_metrics_aggregation_fn']
        if 'fit_metrics_aggregation_fn' in kwargs:
            fedavg_params['fit_metrics_aggregation_fn'] = kwargs['fit_metrics_aggregation_fn']

        # Initialize standard FedAvg strategy for basic aggregation
        self.fedavg_strategy = FedAvg(**fedavg_params)

        # Initialize additional components for enhanced functionality
        self.current_round = 0

        # Load server-side validation dataset for global evaluation
        self._load_server_validation_data()

        # Initialize SecAgg+ workflow
        self.secagg_workflow = SecAggPlusWorkflow(
            num_shares=self.num_shares,
            reconstruction_threshold=self.reconstruction_threshold,
            max_weight=self.max_weight,
            timeout=self.timeout,
            clipping_range=self.clipping_range,
            quantization_range=self.quantization_range,
        )

        print(f"SecAgg+ initialized with parameters:")
        print(f"  - num_shares: {self.num_shares}")
        print(f"  - reconstruction_threshold: {self.reconstruction_threshold}")
        print(f"  - max_weight: {self.max_weight}")
        print(f"  - timeout: {self.timeout}")
        print(f"  - clipping_range: {self.clipping_range}")
        print(f"  - quantization_range: {self.quantization_range}")

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "secagg+"

    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy-specific parameters."""
        return {
            "num_shares": self.num_shares,
            "reconstruction_threshold": self.reconstruction_threshold,
            "max_weight": self.max_weight,
            "timeout": self.timeout,
            "clipping_range": self.clipping_range,
            "quantization_range": self.quantization_range,
            "fraction_fit": self.fedavg_strategy.fraction_fit,
            "fraction_evaluate": self.fedavg_strategy.fraction_evaluate,
            "min_fit_clients": self.fedavg_strategy.min_fit_clients,
            "min_evaluate_clients": self.fedavg_strategy.min_evaluate_clients,
            "min_available_clients": self.fedavg_strategy.min_available_clients,
        }

    def get_secagg_workflow(self) -> SecAggPlusWorkflow:
        """Get the SecAgg+ workflow for use in server applications.

        Returns:
            SecAggPlusWorkflow: The configured SecAgg+ workflow
        """
        return self.secagg_workflow

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate fit results using SecAgg+ with enhanced logging.

        Note: This method is called by the DefaultWorkflow when using SecAggPlusWorkflow.
        The actual secure aggregation is handled by the SecAggPlusWorkflow.
        """
        self.current_round = server_round

        print(f"SecAgg+ aggregation for round {server_round} with {len(results)} clients")

        # Log client training metrics
        if self.wandb_logger:
            for client_proxy, fit_res in results:
                client_metrics = fit_res.metrics
                if client_metrics:
                    client_id = fit_res.metrics.get("client_id", "unknown")
                    metrics_to_log = {k: v for k, v in client_metrics.items() if k != "client_id"}
                    self.wandb_logger.log_metrics(
                        metrics_to_log,
                        prefix=f"client_{client_id}/fit",
                        step=server_round
                    )

        # Use the standard FedAvg aggregation for post-processing
        # (The actual SecAgg+ aggregation is handled by the workflow)
        aggregated_parameters, metrics = self.fedavg_strategy.aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is None:
            return None, {}

        # Update server model with aggregated parameters
        param_arrays = safe_parameters_to_ndarrays(aggregated_parameters)
        set_params(self.model, param_arrays)

        # Collect metrics with SecAgg+ specific information
        secagg_metrics = {
            "num_clients": len(results),
            "num_failures": len(failures),
            "num_shares": self.num_shares,
            "reconstruction_threshold": self.reconstruction_threshold,
            "max_weight": self.max_weight,
            "clipping_range": self.clipping_range,
            "quantization_range": self.quantization_range,
        }

        # Merge with existing metrics
        if metrics:
            secagg_metrics.update(metrics)

        # Log aggregated fit metrics with SecAgg+ specific information
        if self.wandb_logger and secagg_metrics:
            self.wandb_logger.log_metrics(
                secagg_metrics,
                prefix="server",
                step=server_round
            )

        # Print server model's current metrics
        print(f"Server model metrics after round {server_round} (SecAgg+):")
        for metric_name, metric_value in secagg_metrics.items():
            print(f"  {metric_name}: {metric_value}")

        # Save frequency checkpoint
        if self.config.training.checkpoint.save_regular and server_round % self.config.training.checkpoint.save_frequency == 0:
            self._save_checkpoint(self.model.state_dict(), server_round)

        return aggregated_parameters, secagg_metrics

    # Implement required Strategy abstract methods
    def initialize_parameters(self, client_manager):
        """Initialize global model parameters."""
        print("SecAgg+ Strategy: Initializing parameters from server model")
        ndarrays = get_params(self.model)
        print(f"SecAgg+ Strategy: Sending {len(ndarrays)} parameter arrays to clients")
        print(f"SecAgg+ Strategy: First few parameter shapes: {[arr.shape for arr in ndarrays[:5]]}")
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get the base configuration from the parent class
        client_instructions = self.fedavg_strategy.configure_fit(
            server_round, parameters, client_manager
        )

        # Add server_round and SecAgg+ parameters to the config for each client
        updated_instructions = []
        for client_proxy, fit_ins in client_instructions:
            # Add server_round and SecAgg+ parameters to the existing config
            config = fit_ins.config.copy() if fit_ins.config else {}
            config["server_round"] = server_round
            config["secagg_enabled"] = True
            config["num_shares"] = self.num_shares
            config["reconstruction_threshold"] = self.reconstruction_threshold

            # Create new FitIns with updated config
            updated_fit_ins = FitIns(
                parameters=fit_ins.parameters,
                config=config
            )
            updated_instructions.append((client_proxy, updated_fit_ins))

        # Add WandB run ID to instructions
        return self.add_wandb_config_to_instructions(updated_instructions)

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Get the base configuration from the parent class
        client_instructions = self.fedavg_strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

        # Add server_round to the config for each client
        updated_instructions = []
        for client_proxy, evaluate_ins in client_instructions:
            # Add server_round to the existing config
            config = evaluate_ins.config.copy() if evaluate_ins.config else {}
            config["server_round"] = server_round

            # Create new EvaluateIns with updated config
            updated_evaluate_ins = EvaluateIns(
                parameters=evaluate_ins.parameters,
                config=config
            )
            updated_instructions.append((client_proxy, updated_evaluate_ins))

        # Add WandB run ID to instructions
        return self.add_wandb_config_to_instructions(updated_instructions)

    def get_strategy_specific_metrics(self) -> Dict[str, Any]:
        """Return strategy-specific metrics for SecAgg+.

        Returns:
            Dictionary of SecAgg+ specific metrics for logging.
        """
        return {
            "num_shares": self.num_shares,
            "reconstruction_threshold": self.reconstruction_threshold,
            "secagg_enabled": True,
        }

    def evaluate(self, server_round, parameters):
        """Evaluate model parameters."""
        return self.fedavg_strategy.evaluate(server_round, parameters)

    # Delegate other attributes to FedAvg
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying FedAvg strategy."""
        return getattr(self.fedavg_strategy, name)


class SecAggPlusClient(ClientStrategyBase):
    """Client-side Real Secure Aggregation (SecAgg+) strategy using Flower's built-in SecAgg."""

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
        """Initialize SecAgg+ client strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to use for computation
            scheduler: Learning rate scheduler (optional)
            **kwargs: Additional strategy parameters
        """
        super().__init__(config, model, optimizer, criterion, device, scheduler, **kwargs)

        # Client ID must be explicitly set - FAIL FAST if not specified
        if not hasattr(config.fl, 'client_id') or config.fl.client_id is None:
            raise ValueError(
                "ERROR: 'client_id' not specified in client config. "
                "You must explicitly set 'client_id' in the FL config section. "
                "This prevents client identification issues in federated learning."
            )
        self.client_id = config.fl.client_id
        self.current_round = 0

        # SecAgg+ specific parameters
        self.mixed_precision = config.training.mixed_precision
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        print(f"SecAgg+ Client initialized for client_id: {self.client_id}")

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "secagg+"

    def prepare_for_round(self, server_params: Parameters, round_config: Dict[str, Any]):
        """Prepare the client for a new training round.

        Args:
            server_params: Parameters from server
            round_config: Configuration for this round
        """
        # Convert parameters to numpy arrays safely
        param_arrays = safe_parameters_to_ndarrays(server_params)

        # Update model with server parameters
        set_params(self.model, param_arrays)

        # Update round number
        self.current_round = round_config.get("server_round", self.current_round + 1)

        # Store current FL round information
        self.current_fl_round = round_config.get("server_round", 1)

        # Reset optimizer state
        self.optimizer.zero_grad()

        # Check if SecAgg is enabled
        secagg_enabled = round_config.get("secagg_enabled", False)
        if secagg_enabled:
            print(f"SecAgg+ enabled for client {self.client_id}, round {self.current_round}")
        else:
            print(f"Warning: SecAgg+ not enabled for client {self.client_id}, round {self.current_round}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        **kwargs
    ) -> Tuple[float, float]:
        """Train the model for one epoch using SecAgg+.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            total_epochs: Total number of epochs
            **kwargs: Additional training parameters

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Mixed precision training
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

        # Step the scheduler only once per FL round (after the last local epoch)
        if self.scheduler is not None and epoch == total_epochs - 1:  # Only on last local epoch
            current_lr_before = self.optimizer.param_groups[0]['lr']

            # Handle ReduceLROnPlateau scheduler which requires validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)  # Use training loss as proxy
            else:
                self.scheduler.step()

            current_lr_after = self.optimizer.param_groups[0]['lr']
            if current_lr_before != current_lr_after:
                print(f"FL Round {getattr(self, 'current_fl_round', '?')}: LR changed from {current_lr_before:.8f} to {current_lr_after:.8f}")

        return avg_loss, avg_accuracy

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters for SecAgg+ transmission.

        Note: The actual secure aggregation masking is handled by the secaggplus_mod.
        We just return the raw parameters here.

        Returns:
            List of parameter arrays
        """
        return get_params(self.model)

    def get_custom_metrics(self) -> Dict[str, Any]:
        """Return custom SecAgg+ specific metrics.

        Returns:
            Dictionary of custom metrics
        """
        return {
            "secagg_enabled": True,
            "client_id": self.client_id,
        }


class SecAggPlusFlowerClient(NumPyClient):
    """Flower NumPyClient implementation for SecAgg+ that integrates with existing ADNI architecture."""

    def __init__(self, client_strategy: SecAggPlusClient, train_loader: DataLoader, val_loader: DataLoader):
        """Initialize the Flower client.

        Args:
            client_strategy: SecAgg+ client strategy instance
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.client_strategy = client_strategy
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters."""
        return self.client_strategy.get_parameters()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train the model and return updated parameters."""
        # Convert parameters to Flower format
        params = ndarrays_to_parameters(parameters)

        # Prepare for training round
        self.client_strategy.prepare_for_round(params, config)

        # Train for specified number of epochs
        num_epochs = config.get("local_epochs", 1)
        total_loss = 0.0
        total_accuracy = 0.0

        for epoch in range(num_epochs):
            loss, accuracy = self.client_strategy.train_epoch(
                self.train_loader, epoch, num_epochs
            )
            total_loss += loss
            total_accuracy += accuracy

        avg_loss = total_loss / num_epochs
        avg_accuracy = total_accuracy / num_epochs

        # Get updated parameters
        updated_params = self.client_strategy.get_parameters()

        # Get custom metrics
        custom_metrics = self.client_strategy.get_custom_metrics()
        custom_metrics.update({
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
        })

        return updated_params, len(self.train_loader.dataset), custom_metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model."""
        # Convert parameters to Flower format and set model parameters
        params = ndarrays_to_parameters(parameters)
        param_arrays = safe_parameters_to_ndarrays(params)
        set_params(self.client_strategy.model, param_arrays)

        # Evaluate model
        val_loss, val_accuracy, predictions, labels = test_with_predictions(
            model=self.client_strategy.model,
            test_loader=self.val_loader,
            criterion=self.client_strategy.criterion,
            device=self.client_strategy.device,
            mixed_precision=self.client_strategy.mixed_precision
        )

        # Prepare metrics
        metrics = {
            "accuracy": val_accuracy,
            "client_id": self.client_strategy.client_id,
        }

        # Add predictions and labels for confusion matrix if available
        if predictions is not None and labels is not None:
            import json
            metrics["predictions_json"] = json.dumps(predictions)
            metrics["labels_json"] = json.dumps(labels)

        return val_loss, len(self.val_loader.dataset), metrics

    def to_client(self):
        """Convert to Flower client format."""
        return self


def create_secagg_plus_client_fn(config: Config, model: nn.Module, optimizer: torch.optim.Optimizer,
                                criterion: nn.Module, device: torch.device,
                                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    """Create a client function for SecAgg+ that can be used with Flower's ClientApp.

    Args:
        config: Configuration object
        model: PyTorch model
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to use for computation
        scheduler: Learning rate scheduler (optional)

    Returns:
        Client function compatible with Flower's ClientApp
    """
    def client_fn(context: Context):
        """Client function that creates a SecAgg+ client."""
        # Load data for this client
        train_loader, val_loader = load_data(
            config=config,
            batch_size=config.training.batch_size
        )

        # Create SecAgg+ client strategy
        client_strategy = SecAggPlusClient(
            config=config,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler
        )

        # Create and return Flower client
        return SecAggPlusFlowerClient(client_strategy, train_loader, val_loader)

    return client_fn
