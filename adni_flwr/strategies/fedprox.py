"""FedProx strategy implementation."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.common import Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters, FitIns, EvaluateIns
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .base import FLStrategyBase, ClientStrategyBase
from adni_classification.config.config import Config
from adni_flwr.task import set_params, get_params, safe_parameters_to_ndarrays, load_data, test_with_predictions, create_criterion
from adni_classification.utils.visualization import plot_confusion_matrix

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class FedProxStrategy(FLStrategyBase):
    """Server-side FedProx strategy with comprehensive WandB logging."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        wandb_logger: Optional[Any] = None,
        mu: float = 0.01,
        **kwargs
    ):
        """Initialize FedProx strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            wandb_logger: Wandb logger instance
            mu: FedProx regularization parameter
            **kwargs: Additional FedProx parameters
        """
        super().__init__(config, model, wandb_logger, **kwargs)

        self.mu = mu

        # Extract specific parameters for FedAvg (FedProx uses same aggregation)
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

        # Initialize standard FedAvg strategy for aggregation
        self.fedavg_strategy = FedAvg(**fedavg_params)

        # Initialize additional components for enhanced functionality
        self.current_round = 0

        # Load server-side validation dataset for global evaluation
        self._load_server_validation_data()

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "fedprox"

    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy-specific parameters."""
        return {
            "mu": self.mu,
            "fraction_fit": self.fedavg_strategy.fraction_fit,
            "fraction_evaluate": self.fedavg_strategy.fraction_evaluate,
            "min_fit_clients": self.fedavg_strategy.min_fit_clients,
            "min_evaluate_clients": self.fedavg_strategy.min_evaluate_clients,
            "min_available_clients": self.fedavg_strategy.min_available_clients,
        }

    def get_strategy_specific_metrics(self) -> Dict[str, Any]:
        """Return strategy-specific metrics for FedProx.

        Returns:
            Dictionary of FedProx specific metrics for logging.
        """
        return {
            "fedprox_mu": self.mu,
        }

    # Implement required Strategy abstract methods (FedProx uses same aggregation as FedAvg)
    def initialize_parameters(self, client_manager):
        """Initialize global model parameters."""
        # Instead of delegating to fedavg_strategy (which returns None),
        # provide initial parameters from our server model
        from adni_flwr.task import get_params
        from flwr.common import ndarrays_to_parameters

        print("FedProxStrategy: Initializing parameters from server model")
        ndarrays = get_params(self.model)
        print(f"FedProxStrategy: Sending {len(ndarrays)} parameter arrays to clients")
        print(f"FedProxStrategy: First few parameter shapes: {[arr.shape for arr in ndarrays[:5]]}")

        return ndarrays_to_parameters(ndarrays)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get the base configuration from the parent class
        client_instructions = self.fedavg_strategy.configure_fit(
            server_round, parameters, client_manager
        )

        # Add server_round and FedProx mu to the config for each client
        updated_instructions = []
        for client_proxy, fit_ins in client_instructions:
            # Add server_round and mu to the existing config
            config = fit_ins.config.copy() if fit_ins.config else {}
            config["server_round"] = server_round
            config["mu"] = self.mu

            # Create new FitIns with updated config
            updated_fit_ins = FitIns(
                parameters=fit_ins.parameters,
                config=config
            )
            updated_instructions.append((client_proxy, updated_fit_ins))

        # Add WandB run ID to instructions
        return self.add_wandb_config_to_instructions(updated_instructions)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate fit results using FedProx with enhanced logging."""
        self.current_round = server_round

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

        # Aggregate parameters and metrics using base FedAvg
        aggregated_params, aggregated_metrics = self.fedavg_strategy.aggregate_fit(
            server_round, results, failures
        )

        # Load aggregated parameters into the server model instance
        if aggregated_params is not None:
            param_arrays = safe_parameters_to_ndarrays(aggregated_params)
            set_params(self.model, param_arrays)

        # Collect metrics with FedProx-specific information
        metrics = {
            "num_clients": len(results),
            "num_failures": len(failures),
            "fedprox_mu": self.mu,
        }

        # Log aggregated fit metrics with FedProx-specific information
        if self.wandb_logger and metrics:
            self.wandb_logger.log_metrics(
                metrics,
                prefix="server",
                step=server_round
            )

        # Print server model's current metrics
        print(f"Server model metrics after round {server_round} (FedProx mu={self.mu}):")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")

        # Save frequency checkpoint
        if server_round % self.config.training.checkpoint.save_frequency == 0:
            self._save_checkpoint(self.model.state_dict(), server_round)

        return aggregated_params, metrics

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

    def evaluate(self, server_round, parameters):
        """Evaluate model parameters."""
        return self.fedavg_strategy.evaluate(server_round, parameters)

    # Delegate other attributes to FedAvg
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying FedAvg strategy."""
        return getattr(self.fedavg_strategy, name)


class FedProxClient(ClientStrategyBase):
    """Client-side FedProx strategy with proximal regularization."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        mu: float = 0.01,
        **kwargs
    ):
        """Initialize FedProx client strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to use for computation
            scheduler: Learning rate scheduler (optional)
            mu: FedProx regularization parameter
            **kwargs: Additional strategy parameters
        """
        super().__init__(config, model, optimizer, criterion, device, scheduler, **kwargs)

        self.mu = mu
        self.global_params = None  # Store global model parameters for proximal term

        # FedProx-specific parameters
        self.mixed_precision = config.training.mixed_precision
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        print(f"FedProxClient initialized with mu={self.mu}")

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "fedprox"

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

        # Store global model parameters for proximal term
        self.global_params = {}
        for name, param in self.model.named_parameters():
            self.global_params[name] = param.clone().detach()

        # Store current FL round information
        self.current_fl_round = round_config.get("server_round", 1)

        # Reset optimizer state
        self.optimizer.zero_grad()

        # Update mu if specified in round config
        if "fedprox_mu" in round_config:
            self.mu = round_config["fedprox_mu"]
            print(f"Updated mu to {self.mu} for this round")

    def compute_proximal_loss(self) -> torch.Tensor:
        """Compute the proximal regularization term.

        Returns:
            Proximal loss tensor
        """
        proximal_loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.global_params:
                global_param = self.global_params[name].to(self.device)
                proximal_loss = proximal_loss + torch.norm(param - global_param) ** 2

        return (self.mu / 2.0) * proximal_loss

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        **kwargs
    ) -> Tuple[float, float]:
        """Train the model for one epoch using FedProx.

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
        total_proximal_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Mixed precision training
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    classification_loss = self.criterion(outputs, labels)
                    proximal_loss = self.compute_proximal_loss()
                    total_loss_batch = classification_loss + proximal_loss
                    total_loss_batch = total_loss_batch / self.gradient_accumulation_steps

                self.scaler.scale(total_loss_batch).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                classification_loss = self.criterion(outputs, labels)
                proximal_loss = self.compute_proximal_loss()
                total_loss_batch = classification_loss + proximal_loss
                total_loss_batch = total_loss_batch / self.gradient_accumulation_steps
                total_loss_batch.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            total_loss += classification_loss.item() * self.gradient_accumulation_steps
            total_proximal_loss += proximal_loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        avg_proximal_loss = total_proximal_loss / len(train_loader)
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

        print(f"  Epoch {epoch+1}/{total_epochs}: "
              f"classification_loss={avg_loss:.4f}, "
              f"proximal_loss={avg_proximal_loss:.4f}, "
              f"accuracy={avg_accuracy:.2f}%")

        return avg_loss, avg_accuracy

    def get_custom_metrics(self) -> Dict[str, Any]:
        """Return custom FedProx-specific metrics.

        Returns:
            Dictionary of custom metrics
        """
        # Return FedProx-specific metrics for WandB tracking
        return {
            "fedprox_mu": self.mu
        }

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return FedProx-specific checkpoint data.

        Returns:
            Dictionary containing global model parameters and mu value
        """
        return {
            "mu": self.mu,
            "global_model_params": self.global_params
        }

    def load_checkpoint_data(self, checkpoint_data: Dict[str, Any]):
        """Load FedProx-specific checkpoint data.

        Args:
            checkpoint_data: FedProx-specific checkpoint data
        """
        if "mu" in checkpoint_data:
            self.mu = checkpoint_data["mu"]
            print(f"Restored FedProx mu parameter: {self.mu}")

        if "global_model_params" in checkpoint_data:
            self.global_params = checkpoint_data["global_model_params"]
            if self.global_params:
                print(f"Restored FedProx global model parameters for {len(self.global_params)} layers")
            else:
                print("Restored FedProx global model parameters: None")
