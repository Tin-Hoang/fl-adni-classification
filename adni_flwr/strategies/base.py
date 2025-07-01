"""Base classes for Federated Learning strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import os
import pickle
import random
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from flwr.common import Parameters, FitRes, EvaluateRes, ConfigRecord, ArrayRecord, Array
from flwr.server.strategy import Strategy
from flwr.client import NumPyClient
import gc

from adni_classification.config.config import Config
from adni_classification.utils.training_utils import get_scheduler
from adni_flwr.task import (
    load_data,
    safe_parameters_to_ndarrays,
    set_params,
    test_with_predictions,
    is_fl_client_checkpoint
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

    def get_wandb_run_id(self) -> Optional[str]:
        """Get the WandB run ID for sharing with clients.

        Returns:
            WandB run ID if available, None otherwise
        """
        if self.wandb_logger and hasattr(self.wandb_logger, 'get_run_id'):
            return self.wandb_logger.get_run_id()
        return None

    def add_wandb_config_to_instructions(self, instructions: List[Tuple[Any, Any]], config_key: str = "config") -> List[Tuple[Any, Any]]:
        """Add WandB run ID to client instructions.

        Args:
            instructions: List of (client_proxy, instruction) tuples
            config_key: Key to access config in instruction (usually "config")

        Returns:
            Updated instructions with WandB run ID
        """
        run_id = self.get_wandb_run_id()
        if not run_id:
            return instructions

        updated_instructions = []
        for client_proxy, instruction in instructions:
            # Get the existing config
            if hasattr(instruction, config_key):
                config = getattr(instruction, config_key)
                config = config.copy() if config else {}
            else:
                config = {}

            # Add WandB run ID
            config["wandb_run_id"] = run_id

            # Create updated instruction
            if hasattr(instruction, '_replace'):
                # For named tuples like FitIns/EvaluateIns
                updated_instruction = instruction._replace(config=config)
            else:
                # For other instruction types, try to update config attribute
                updated_instruction = instruction
                setattr(updated_instruction, config_key, config)

            updated_instructions.append((client_proxy, updated_instruction))

        return updated_instructions

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
        client_strategy: ClientStrategyBase,
        context=None,
        total_fl_rounds: int = None,
        wandb_logger=None
    ):
        """Initialize strategy-aware client.

        Args:
            config: Configuration object
            device: Device to use for computation
            client_strategy: Client-side strategy implementation
            context: Flower Context for stateful client management (optional)
            total_fl_rounds: Total number of FL rounds for scheduler initialization
            wandb_logger: Client-side WandB logger for distributed training (optional)
        """
        self.config = config
        self.device = device
        self.client_strategy = client_strategy
        self.context = context
        self.total_fl_rounds = total_fl_rounds
        self.wandb_logger = wandb_logger
        # Client ID must be explicitly set - FAIL FAST if not specified
        if not hasattr(config.fl, 'client_id') or config.fl.client_id is None:
            raise ValueError(
                "ERROR: 'client_id' not specified in client config. "
                "You must explicitly set 'client_id' in the FL config section. "
                "This prevents client identification issues in federated learning."
            )
        self.client_id = config.fl.client_id

        # Load data
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

        # Initialize Context-based scheduler management if context provided
        if self.context is not None and self.total_fl_rounds is not None:
            self._initialize_context_scheduler()

        # MEMORY FIX: Make training_history optional since WandB already tracks everything
        self.enable_local_history = getattr(config.training, 'enable_local_history', False)
        if self.enable_local_history:
            print(f"Client {self.client_id}: Local training history enabled (memory usage will increase)")
            self.training_history = {
                'train_losses': [],
                'train_accuracies': [],
                'val_losses': [],
                'val_accuracies': [],
                'rounds': []
            }
        else:
            print(f"Client {self.client_id}: Local training history disabled (using WandB tracking only)")
            self.training_history = None

        print(f"Initialized {self.client_strategy.get_strategy_name()} client with config: {self.config.wandb.run_name if hasattr(self.config, 'wandb') and hasattr(self.config.wandb, 'run_name') else 'unknown'}")
        print(f"Train dataset size: {len(self.train_loader.dataset)}")
        print(f"Validation dataset size: {len(self.val_loader.dataset)}")
        print(f"Evaluation frequency: every {self.evaluate_frequency} round(s)")
        print(f"Client checkpoint directory: {self.client_checkpoint_dir}")
        print(f"Client checkpoint saving: {'enabled' if self.save_client_checkpoints else 'disabled'} (frequency: {self.checkpoint_save_frequency})")

    def _initialize_context_scheduler(self):
        """Initialize or restore scheduler from lightweight context tracking."""
        # Initialize context state for lightweight scheduler tracking only
        if "scheduler_tracking" not in self.context.state.config_records:
            self.context.state.config_records["scheduler_tracking"] = ConfigRecord()

        scheduler_tracking = self.context.state.config_records["scheduler_tracking"]

        if "scheduler_type" not in scheduler_tracking:
            # First time - create fresh scheduler and store lightweight tracking info
            print(f"Creating new scheduler '{self.config.training.lr_scheduler}' for {self.total_fl_rounds} FL rounds")

            # Always recreate optimizer fresh (memory efficient)
            self._recreate_optimizer()

            scheduler = get_scheduler(
                scheduler_type=self.config.training.lr_scheduler,
                optimizer=self.client_strategy.optimizer,
                num_epochs=self.total_fl_rounds
            )

            if scheduler is not None:
                # Store only lightweight scheduler tracking info
                scheduler_tracking["scheduler_type"] = self.config.training.lr_scheduler
                scheduler_tracking["total_rounds"] = self.total_fl_rounds
                scheduler_tracking["last_epoch"] = -1  # Starting position
                scheduler_tracking["base_lr"] = self.config.training.learning_rate

                # Store scheduler-specific parameters
                if hasattr(scheduler, 'T_max'):
                    scheduler_tracking["T_max"] = scheduler.T_max
                if hasattr(scheduler, 'eta_min'):
                    scheduler_tracking["eta_min"] = scheduler.eta_min
                if hasattr(scheduler, 'step_size'):
                    scheduler_tracking["step_size"] = scheduler.step_size
                if hasattr(scheduler, 'gamma'):
                    scheduler_tracking["gamma"] = scheduler.gamma

                self.client_strategy.scheduler = scheduler
                print(f"Lightweight scheduler tracking initialized: {dict(scheduler_tracking)}")
            else:
                print("No scheduler specified")
                self.client_strategy.scheduler = None
        else:
            # Restore scheduler from lightweight tracking
            print(f"Restoring scheduler from lightweight tracking")

            # Always recreate optimizer fresh (memory efficient)
            self._recreate_optimizer()

            # Recreate scheduler with tracked state
            scheduler_type = scheduler_tracking.get("scheduler_type", self.config.training.lr_scheduler)
            last_epoch = scheduler_tracking.get("last_epoch", -1)

            scheduler = self._recreate_scheduler_from_tracking(scheduler_tracking)

            if scheduler is not None:
                self.client_strategy.scheduler = scheduler
                current_lr = self.client_strategy.optimizer.param_groups[0]['lr']
                print(f"Scheduler ({type(scheduler).__name__}) recreated from lightweight tracking")
                print(f"Restored to last_epoch={last_epoch}, current LR: {current_lr:.8f}")

                # Print scheduler diagnostics
                if hasattr(scheduler, 'get_lr'):
                    try:
                        expected_lrs = scheduler.get_lr()
                        print(f"Scheduler's expected LR: {[f'{lr:.8f}' for lr in expected_lrs]}")

                        # CRITICAL FIX: Synchronize optimizer LR with scheduler's expected LR
                        if last_epoch > -1:
                            print(f"Synchronizing optimizer LR with scheduler expectation after restoration")
                            self._synchronize_scheduler_lr(scheduler, last_epoch)

                            # Show final LR after synchronization
                            final_lr = self.client_strategy.optimizer.param_groups[0]['lr']
                            print(f"Final synchronized LR: {final_lr:.8f}")

                    except Exception as lr_check_error:
                        print(f"Could not check scheduler's expected LR: {lr_check_error}")
            else:
                print("Failed to recreate scheduler from lightweight tracking")
                self.client_strategy.scheduler = None

    def _recreate_optimizer(self):
        """Recreate optimizer fresh for memory efficiency."""
        current_params = list(self.client_strategy.model.parameters())

        # Create fresh optimizer based on config
        if hasattr(self.config.training, 'optimizer'):
            optimizer_type = self.config.training.optimizer.lower()
        else:
            optimizer_type = 'adamw'  # default

        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay

        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(current_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(current_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = getattr(self.config.training, 'momentum', 0.9)
            optimizer = torch.optim.SGD(current_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            # Default to AdamW
            optimizer = torch.optim.AdamW(current_params, lr=lr, weight_decay=weight_decay)

        self.client_strategy.optimizer = optimizer
        print(f"Recreated fresh {type(optimizer).__name__} optimizer with LR={lr:.8f}")

    def _recreate_scheduler_from_tracking(self, scheduler_tracking) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Recreate scheduler from lightweight tracking information."""
        try:
            scheduler_type = scheduler_tracking.get("scheduler_type")
            last_epoch = scheduler_tracking.get("last_epoch", -1)

            if scheduler_type == "CosineAnnealingLR":
                T_max = scheduler_tracking.get("T_max", self.total_fl_rounds)
                eta_min = scheduler_tracking.get("eta_min", 0.0)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.client_strategy.optimizer,
                    T_max=T_max,
                    eta_min=eta_min,
                    last_epoch=last_epoch
                )
            elif scheduler_type == "StepLR":
                step_size = scheduler_tracking.get("step_size", 30)
                gamma = scheduler_tracking.get("gamma", 0.1)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    self.client_strategy.optimizer,
                    step_size=step_size,
                    gamma=gamma,
                    last_epoch=last_epoch
                )
            elif scheduler_type == "MultiStepLR":
                milestones = scheduler_tracking.get("milestones", [30, 60, 90])
                gamma = scheduler_tracking.get("gamma", 0.1)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.client_strategy.optimizer,
                    milestones=milestones,
                    gamma=gamma,
                    last_epoch=last_epoch
                )
            elif scheduler_type == "ExponentialLR":
                gamma = scheduler_tracking.get("gamma", 0.95)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.client_strategy.optimizer,
                    gamma=gamma,
                    last_epoch=last_epoch
                )
            else:
                # Fallback using get_scheduler
                scheduler = get_scheduler(
                    scheduler_type=scheduler_type,
                    optimizer=self.client_strategy.optimizer,
                    num_epochs=self.total_fl_rounds
                )
                if scheduler is not None and last_epoch > -1:
                    # Manually set last_epoch for unknown scheduler types
                    scheduler.last_epoch = last_epoch

            return scheduler

        except Exception as e:
            print(f"Error recreating scheduler from tracking: {e}")
            return None

    def _update_context_scheduler(self):
        """Update lightweight scheduler tracking in context after training."""
        if self.context is not None and self.client_strategy.scheduler is not None:
            scheduler_tracking = self.context.state.config_records["scheduler_tracking"]

            try:
                # Update only the lightweight tracking information
                if hasattr(self.client_strategy.scheduler, 'last_epoch'):
                    current_step = self.client_strategy.scheduler.last_epoch
                else:
                    current_step = getattr(self.client_strategy.scheduler, '_step_count', 0)

                scheduler_tracking["last_epoch"] = current_step

                print(f"Updated lightweight scheduler tracking: last_epoch = {current_step}")

            except Exception as e:
                print(f"ERROR: Failed to update scheduler tracking: {e}")

    def _synchronize_scheduler_lr(self, scheduler, current_step: int):
        """Synchronize optimizer learning rate with scheduler expectations.

        This is crucial when the scheduler is restored from lightweight tracking
        but the optimizer is fresh, which may cause LR mismatch.

        Args:
            scheduler: The scheduler instance
            current_step: Current training step
        """
        try:
            # For different scheduler types, we need different approaches
            scheduler_type = type(scheduler).__name__

            if scheduler_type == "CosineAnnealingLR":
                # For CosineAnnealingLR, we can calculate the expected LR
                if hasattr(scheduler, 'get_lr'):
                    calculated_lrs = scheduler.get_lr()
                    for param_group, lr in zip(scheduler.optimizer.param_groups, calculated_lrs):
                        old_lr = param_group['lr']
                        param_group['lr'] = lr
                        print(f"LR sync: Updated param group LR from {old_lr:.8f} to {lr:.8f}")

                    # Update scheduler's internal _last_lr if it exists
                    if hasattr(scheduler, '_last_lr'):
                        scheduler._last_lr = calculated_lrs

                    print(f"Successfully synchronized LR for {scheduler_type} at step {current_step}")
                else:
                    print(f"Warning: {scheduler_type} scheduler doesn't have get_lr() method")

            elif scheduler_type in ["StepLR", "MultiStepLR", "ExponentialLR"]:
                # For step-based schedulers, calculate expected LR
                if hasattr(scheduler, 'get_lr'):
                    calculated_lrs = scheduler.get_lr()
                    for param_group, lr in zip(scheduler.optimizer.param_groups, calculated_lrs):
                        old_lr = param_group['lr']
                        param_group['lr'] = lr
                        print(f"LR sync: Updated param group LR from {old_lr:.8f} to {lr:.8f}")

                    if hasattr(scheduler, '_last_lr'):
                        scheduler._last_lr = calculated_lrs

                    print(f"Successfully synchronized LR for {scheduler_type} at step {current_step}")

            elif scheduler_type == "ReduceLROnPlateau":
                # ReduceLROnPlateau doesn't have get_lr(), and its LR changes are event-driven
                # We can't easily predict the LR without knowing the loss history
                print(f"Warning: Cannot synchronize LR for {scheduler_type} - LR changes are loss-driven")

            else:
                print(f"Warning: Unknown scheduler type {scheduler_type}, attempting generic LR sync")
                if hasattr(scheduler, 'get_lr'):
                    calculated_lrs = scheduler.get_lr()
                    for param_group, lr in zip(scheduler.optimizer.param_groups, calculated_lrs):
                        old_lr = param_group['lr']
                        param_group['lr'] = lr
                        print(f"LR sync: Updated param group LR from {old_lr:.8f} to {lr:.8f}")

                    if hasattr(scheduler, '_last_lr'):
                        scheduler._last_lr = calculated_lrs

        except Exception as e:
            print(f"Error during LR synchronization: {e}")
            print(f"Continuing with current LR settings")

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

        # Check scheduler health before attempting to save
        scheduler_healthy = self._check_scheduler_health()
        if not scheduler_healthy:
            print(f"Client {self.client_id}: WARNING - Scheduler health check failed, checkpoint may fail")

        try:
            checkpoint = {
                'round': round_num,
                'client_id': self.client_id,
                'model_state_dict': self.client_strategy.model.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'best_val_accuracy': self.best_val_accuracy,
                'strategy_name': self.client_strategy.get_strategy_name(),
                'strategy_metrics': self.client_strategy.get_custom_metrics(),
                'config': {
                    'local_epochs': self.config.fl.local_epochs,
                    'learning_rate': self.config.training.learning_rate,
                    'weight_decay': self.config.training.weight_decay,
                }
            }

            # MEMORY FIX: Only save training_history if enabled
            if self.enable_local_history and self.training_history is not None:
                checkpoint['training_history'] = self.training_history
            else:
                checkpoint['training_history'] = None  # Explicitly set to None to avoid confusion

            # Save optimizer state_dict (simple save since we recreate optimizers fresh each round)
            try:
                checkpoint['optimizer_state_dict'] = self.client_strategy.optimizer.state_dict()
            except Exception as optimizer_error:
                print(f"Client {self.client_id}: Could not save optimizer state: {optimizer_error}")
                # Not critical since we recreate optimizers fresh each round

            # Add scheduler state if available
            if hasattr(self.client_strategy, 'scheduler') and self.client_strategy.scheduler is not None:
                try:
                    checkpoint['scheduler_state_dict'] = self.client_strategy.scheduler.state_dict()
                except Exception as scheduler_error:
                    print(f"Client {self.client_id}: Warning - Could not serialize scheduler state: {scheduler_error}")
                    # Don't include scheduler state in checkpoint if it fails
                    checkpoint['scheduler_serialization_failed'] = True

            # Add strategy-specific checkpoint data
            if hasattr(self.client_strategy, 'get_checkpoint_data'):
                try:
                    checkpoint['strategy_data'] = self.client_strategy.get_checkpoint_data()
                except Exception as strategy_error:
                    print(f"Client {self.client_id}: Warning - Could not get strategy checkpoint data: {strategy_error}")

            # Save regular checkpoint based on frequency
            if round_num % self.checkpoint_save_frequency == 0:
                try:
                    checkpoint_path = os.path.join(self.client_checkpoint_dir, f"checkpoint_round_{round_num}.pt")
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Client {self.client_id}: Saved checkpoint for round {round_num} to {checkpoint_path}")
                except Exception as save_error:
                    print(f"Client {self.client_id}: Error saving round {round_num} checkpoint: {save_error}")

            # Always save latest checkpoint (overwrite)
            try:
                latest_path = os.path.join(self.client_checkpoint_dir, "checkpoint_latest.pt")
                torch.save(checkpoint, latest_path)
            except Exception as save_error:
                print(f"Client {self.client_id}: Error saving latest checkpoint: {save_error}")

            # Save best checkpoint if this is the best model
            if is_best:
                try:
                    best_path = os.path.join(self.client_checkpoint_dir, "checkpoint_best.pt")
                    torch.save(checkpoint, best_path)
                    print(f"Client {self.client_id}: Saved new best checkpoint with accuracy {train_acc:.2f}% to {best_path}")
                except Exception as save_error:
                    print(f"Client {self.client_id}: Error saving best checkpoint: {save_error}")

        except Exception as e:
            print(f"Client {self.client_id}: Error creating checkpoint data: {e}")
            print(f"Client {self.client_id}: Traceback: {traceback.format_exc()}")
        finally:
            # Cleanup memory after checkpoint operations
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as cleanup_error:
                print(f"Client {self.client_id}: Warning - Checkpoint cleanup failed: {cleanup_error}")

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

            # Load optimizer state (simple load since we recreate optimizers fresh each round)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.client_strategy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as optimizer_error:
                    print(f"Client {self.client_id}: Could not load optimizer state: {optimizer_error}")
                    # Not critical since we recreate optimizers fresh each round

            # Load scheduler state if available
            if ('scheduler_state_dict' in checkpoint and
                hasattr(self.client_strategy, 'scheduler') and
                self.client_strategy.scheduler is not None):
                self.client_strategy.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Client {self.client_id}: Restored scheduler state")

            # Restore training state
            self.current_round = checkpoint.get('round', 0)
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)

            # MEMORY FIX: Only restore training_history if local history is enabled
            if self.enable_local_history:
                self.training_history = checkpoint.get('training_history', {
                    'train_losses': [],
                    'train_accuracies': [],
                    'val_losses': [],
                    'val_accuracies': [],
                    'rounds': []
                })
            else:
                # Don't restore training_history if local history is disabled
                self.training_history = None
                print(f"Client {self.client_id}: Skipped loading training_history (local history disabled)")

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
        # Check for WandB run ID in config and join if available
        wandb_run_id = config.get("wandb_run_id")
        if wandb_run_id and self.wandb_logger and not self.wandb_logger.run:
            print(f"Client {self.client_id}: Received WandB run ID from server: {wandb_run_id}")
            self.wandb_logger.join_wandb_run(wandb_run_id)

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
        if self.enable_local_history:
            self.training_history['train_losses'].append(avg_loss)
            self.training_history['train_accuracies'].append(avg_acc)
            self.training_history['rounds'].append(current_round)

        # Check if this is the best training accuracy (simple heuristic)
        is_best = avg_acc > self.best_val_accuracy
        if is_best:
            self.best_val_accuracy = avg_acc

        # Save client checkpoint
        self._save_client_checkpoint(current_round, avg_loss, avg_acc, is_best)

        # Update Context scheduler state if using Context-based management
        if self.context is not None:
            self._update_context_scheduler()

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

        # Log training metrics to WandB if available
        if self.wandb_logger:
            self.wandb_logger.log_metrics(
                metrics,
                prefix="train",
                step=current_round
            )

        # Perform memory cleanup after training
        self._cleanup_memory()

        # MEMORY FIX: Clean up strategy-specific round data if available (e.g., FedProx global params)
        if hasattr(self.client_strategy, 'cleanup_round_data'):
            self.client_strategy.cleanup_round_data()

        # MEMORY FIX: Comprehensive dataset and DataLoader cleanup
        self._cleanup_dataloaders_and_datasets()

        # MEMORY FIX: Simple but effective memory management
        self._simple_memory_cleanup()

        return updated_params, len(self.train_loader.dataset), metrics

    def _cleanup_memory(self):
        """Clean up memory after training to prevent accumulation."""
        try:
            # MEMORY FIX: More thorough cleanup

            # Clear optimizer gradients
            if hasattr(self.client_strategy, 'optimizer') and self.client_strategy.optimizer is not None:
                self.client_strategy.optimizer.zero_grad()

            # Clear model gradients
            if hasattr(self.client_strategy, 'model') and self.client_strategy.model is not None:
                for param in self.client_strategy.model.parameters():
                    if param.grad is not None:
                        param.grad = None

            # Clear PyTorch's CUDA cache if using GPU
            if torch.cuda.is_available():
                # Get memory stats before cleanup
                memory_before = torch.cuda.memory_allocated() / 1024**3  # GB

                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Get memory stats after cleanup
                memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_freed = memory_before - memory_after

                print(f"Client {self.client_id}: GPU memory cleanup - Before: {memory_before:.2f}GB, After: {memory_after:.2f}GB, Freed: {memory_freed:.2f}GB")

            # Force garbage collection
            gc.collect()

            print(f"Client {self.client_id}: Memory cleanup completed")

        except Exception as e:
            print(f"Client {self.client_id}: Warning - Memory cleanup failed: {e}")

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
        # Check for WandB run ID in config and join if available
        wandb_run_id = config.get("wandb_run_id")
        if wandb_run_id and self.wandb_logger and not self.wandb_logger.run:
            print(f"Client {self.client_id}: Received WandB run ID from server: {wandb_run_id}")
            self.wandb_logger.join_wandb_run(wandb_run_id)

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
            if self.enable_local_history:
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

            # Log evaluation metrics to WandB if available
            if self.wandb_logger:
                eval_metrics = {
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "client_id": self.client_id
                }
                self.wandb_logger.log_metrics(
                    eval_metrics,
                    prefix="eval",
                    step=current_round
                )

            # Perform memory cleanup after evaluation
            self._cleanup_memory()

            return float(val_loss), len(self.val_loader.dataset), result

        except Exception as e:
            print(f"Client {self.client_id}: Error in evaluate method: {e}")
            print(traceback.format_exc())

            # Perform memory cleanup even on error
            self._cleanup_memory()

            # Return minimal results to avoid failure
            return 0.0, 0, {
                "client_id": str(self.client_id),
                "error": str(e),
                "evaluation_skipped": False,
                "evaluation_frequency": self.evaluate_frequency,
                "current_round": current_round,
                **self.client_strategy.get_strategy_metrics()
            }

    def _ensure_scheduler_optimizer_sync(self):
        """Ensure scheduler has the correct optimizer reference."""
        if (hasattr(self.client_strategy, 'scheduler') and
            self.client_strategy.scheduler is not None and
            hasattr(self.client_strategy, 'optimizer')):

            # Check if scheduler's optimizer reference matches current optimizer
            if self.client_strategy.scheduler.optimizer is not self.client_strategy.optimizer:
                print(f"Client {self.client_id}: Fixing scheduler optimizer reference mismatch")
                self.client_strategy.scheduler.optimizer = self.client_strategy.optimizer



    def _check_scheduler_health(self) -> bool:
        """Check if scheduler is in a healthy state for serialization.

        Returns:
            True if scheduler can be safely serialized, False otherwise
        """
        if not hasattr(self.client_strategy, 'scheduler') or self.client_strategy.scheduler is None:
            return True  # No scheduler to check

        try:
            # Ensure scheduler has correct optimizer reference
            self._ensure_scheduler_optimizer_sync()

            # Test if scheduler state_dict can be created
            state_dict = self.client_strategy.scheduler.state_dict()

            # Test if we can get basic scheduler info
            scheduler_type = type(self.client_strategy.scheduler).__name__
            last_epoch = getattr(self.client_strategy.scheduler, 'last_epoch', 'N/A')

            # Check optimizer reference
            scheduler_optimizer_id = id(self.client_strategy.scheduler.optimizer)
            current_optimizer_id = id(self.client_strategy.optimizer)
            optimizer_match = scheduler_optimizer_id == current_optimizer_id

            print(f"Client {self.client_id}: Scheduler health check - Type: {scheduler_type}, last_epoch: {last_epoch}, state_dict_keys: {len(state_dict)}, optimizer_ref_match: {optimizer_match}")
            return True

        except Exception as e:
            print(f"Client {self.client_id}: Scheduler health check failed: {e}")
            print(f"Client {self.client_id}: Scheduler health check traceback: {traceback.format_exc()}")
            return False

    def _simple_memory_cleanup(self):
        """Simple but effective memory cleanup to prevent major leaks."""
        try:
            # Clear gradients thoroughly
            if hasattr(self.client_strategy, 'model') and self.client_strategy.model is not None:
                for param in self.client_strategy.model.parameters():
                    if param.grad is not None:
                        param.grad = None

            # Periodically clear optimizer state to prevent accumulation
            if hasattr(self.client_strategy, 'optimizer') and self.current_round % 10 == 0:
                # Store current LR before clearing
                current_lr = self.client_strategy.optimizer.param_groups[0]['lr']
                # Recreate optimizer to clear accumulated state
                self._recreate_optimizer()
                # Restore LR
                for group in self.client_strategy.optimizer.param_groups:
                    group['lr'] = current_lr
                print(f"Client {self.client_id}: Cleared optimizer state at round {self.current_round}")

            # Force garbage collection and CUDA cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Reset peak memory stats periodically
                if self.current_round % 10 == 0:
                    torch.cuda.reset_peak_memory_stats()

            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024**2
                print(f"Client {self.client_id}: Post-cleanup GPU memory: {memory_mb:.1f}MB")

        except Exception as e:
            print(f"Client {self.client_id}: Warning - Simple memory cleanup failed: {e}")

    def _cleanup_dataloaders_and_datasets(self):
        """Clean up DataLoaders and cached datasets to prevent major memory leaks."""
        try:
            # 1. Clean up cached datasets (MAJOR MEMORY LEAK SOURCE)
            if hasattr(self, 'train_loader') and self.train_loader is not None:
                train_dataset = self.train_loader.dataset
                self._cleanup_dataset_cache(train_dataset, "train")

            if hasattr(self, 'val_loader') and self.val_loader is not None:
                val_dataset = self.val_loader.dataset
                self._cleanup_dataset_cache(val_dataset, "val")

            # 2. Clean up DataLoader workers (prevents worker process accumulation)
            if hasattr(self, 'train_loader') and self.train_loader is not None:
                if hasattr(self.train_loader, '_shutdown_workers'):
                    self.train_loader._shutdown_workers()

            if hasattr(self, 'val_loader') and self.val_loader is not None:
                if hasattr(self.val_loader, '_shutdown_workers'):
                    self.val_loader._shutdown_workers()

            print(f"Client {self.client_id}: Cleaned up DataLoaders and datasets")

        except Exception as e:
            print(f"Client {self.client_id}: Warning - DataLoader/dataset cleanup failed: {e}")

    def _cleanup_dataset_cache(self, dataset, dataset_type: str):
        """Clean up cached dataset memory."""
        try:
            # Check if it's a MONAI cached dataset and clear its cache
            if hasattr(dataset, '_cache') and dataset._cache is not None:
                print(f"Client {self.client_id}: Clearing {dataset_type} dataset cache...")

                # For CacheDataset and SmartCacheDataset
                if hasattr(dataset, '_cache'):
                    cache_size = len(dataset._cache) if hasattr(dataset._cache, '__len__') else 'unknown'
                    print(f"  - Clearing cache with {cache_size} items")

                    # Clear the cache
                    if hasattr(dataset._cache, 'clear'):
                        dataset._cache.clear()
                    else:
                        dataset._cache = None

                # For SmartCacheDataset specifically
                if hasattr(dataset, 'cache_data') and dataset.cache_data is not None:
                    print(f"  - Clearing SmartCache cache_data")
                    dataset.cache_data = None

                # Clear any worker pools
                if hasattr(dataset, '_workers') and dataset._workers is not None:
                    print(f"  - Shutting down dataset workers")
                    dataset._workers = None

            # For PersistentDataset, we don't need to clear disk cache, just memory references
            elif hasattr(dataset, 'cache_dir'):
                print(f"Client {self.client_id}: {dataset_type} using PersistentDataset (disk cache) - no memory cleanup needed")

        except Exception as e:
            print(f"Client {self.client_id}: Warning - {dataset_type} dataset cache cleanup failed: {e}")

    def finish_wandb(self):
        """Finish WandB logging for this client."""
        if self.wandb_logger:
            self.wandb_logger.finish()
