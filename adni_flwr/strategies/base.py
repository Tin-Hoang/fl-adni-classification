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
from flwr.common import Parameters, FitRes, EvaluateRes, ConfigRecord
from flwr.server.strategy import Strategy
from flwr.client import NumPyClient

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
        total_fl_rounds: int = None
    ):
        """Initialize strategy-aware client.

        Args:
            config: Configuration object
            device: Device to use for computation
            client_strategy: Client-side strategy implementation
            context: Flower Context for stateful client management (optional)
            total_fl_rounds: Total number of FL rounds for scheduler initialization
        """
        self.config = config
        self.device = device
        self.client_strategy = client_strategy
        self.context = context
        self.total_fl_rounds = total_fl_rounds
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

    def _initialize_context_scheduler(self):
        """Initialize or restore scheduler and optimizer from context state."""
        # Initialize context state for scheduler and optimizer management
        if "scheduler_state" not in self.context.state.config_records:
            self.context.state.config_records["scheduler_state"] = ConfigRecord()
        if "optimizer_state" not in self.context.state.config_records:
            self.context.state.config_records["optimizer_state"] = ConfigRecord()

        scheduler_state = self.context.state.config_records["scheduler_state"]
        optimizer_state = self.context.state.config_records["optimizer_state"]

        # First handle optimizer restoration/initialization
        self._initialize_context_optimizer(optimizer_state)

        # Then handle scheduler (which depends on optimizer)
        if "scheduler" not in scheduler_state:
            # First time - create fresh scheduler
            print(f"Creating new scheduler '{self.config.training.lr_scheduler}' for {self.total_fl_rounds} FL rounds")
            scheduler = get_scheduler(
                scheduler_type=self.config.training.lr_scheduler,
                optimizer=self.client_strategy.optimizer,
                num_epochs=self.total_fl_rounds
            )

            if scheduler is not None:
                # Store scheduler in context state using pickle
                scheduler_state["scheduler"] = pickle.dumps(scheduler)
                scheduler_state["scheduler_type"] = self.config.training.lr_scheduler
                scheduler_state["total_rounds"] = self.total_fl_rounds
                scheduler_state["current_step"] = 0
                self.client_strategy.scheduler = scheduler
                print(f"Scheduler created and stored in context")
            else:
                print("No scheduler specified")
                self.client_strategy.scheduler = None
        else:
            # Restore scheduler from context state
            print(f"Restoring scheduler from context state")
            current_step = scheduler_state.get("current_step", 0)

            try:
                # Load the pickled scheduler directly instead of recreating it
                scheduler = pickle.loads(scheduler_state["scheduler"])

                # Update the scheduler's optimizer reference to the current optimizer
                # This is necessary because the optimizer might be recreated between client sessions
                scheduler.optimizer = self.client_strategy.optimizer

                self.client_strategy.scheduler = scheduler

                # Get the current learning rate after restoration
                current_lr = self.client_strategy.optimizer.param_groups[0]['lr']
                print(f"Scheduler ({type(scheduler).__name__}) successfully restored from pickled state")
                print(f"Restored to step {current_step} (last_epoch={getattr(scheduler, 'last_epoch', 'N/A')})")
                print(f"Current LR after restoration: {current_lr:.8f}")

            except Exception as e:
                print(f"ERROR: Failed to restore pickled scheduler: {e}")
                print(f"Falling back to creating fresh scheduler")

                # Fallback: create fresh scheduler if pickle restoration fails
                scheduler = get_scheduler(
                    scheduler_type=self.config.training.lr_scheduler,
                    optimizer=self.client_strategy.optimizer,
                    num_epochs=self.total_fl_rounds
                )

                if scheduler is not None and current_step > 0:
                    # Apply the old restoration logic as fallback
                    print(f"Fallback: Restoring {type(scheduler).__name__} scheduler with initial LR: {self.client_strategy.optimizer.param_groups[0]['lr']}")
                    scheduler.last_epoch = current_step

                    if hasattr(scheduler, 'get_lr'):
                        calculated_lrs = scheduler.get_lr()
                        for param_group, lr in zip(scheduler.optimizer.param_groups, calculated_lrs):
                            print(f"Fallback: Updating LR from {param_group['lr']:.8f} to {lr:.8f}")
                            param_group['lr'] = lr

                        if hasattr(scheduler, '_last_lr'):
                            scheduler._last_lr = calculated_lrs

                        print(f"Fallback scheduler restoration completed")

                self.client_strategy.scheduler = scheduler

                # Get the current learning rate after fallback restoration
                current_lr = self.client_strategy.optimizer.param_groups[0]['lr']
                print(f"Fallback scheduler restored at step {current_step}, current LR: {current_lr:.8f}")

    def _initialize_context_optimizer(self, optimizer_state):
        """Initialize or restore optimizer from context state."""
        if "optimizer" not in optimizer_state:
            # First time - store the current optimizer
            try:
                # Store optimizer in context state using pickle
                optimizer_pickled = pickle.dumps(self.client_strategy.optimizer)
                optimizer_size = len(optimizer_pickled)

                # Only store if size is reasonable (< 10MB)
                if optimizer_size < 10 * 1024 * 1024:  # 10MB limit
                    optimizer_state["optimizer"] = optimizer_pickled
                    optimizer_state["optimizer_type"] = type(self.client_strategy.optimizer).__name__
                    optimizer_state["size_bytes"] = optimizer_size
                    print(f"Optimizer ({type(self.client_strategy.optimizer).__name__}) stored in context ({optimizer_size:,} bytes)")
                else:
                    print(f"Optimizer too large ({optimizer_size:,} bytes), skipping context storage")
                    optimizer_state["use_state_dict"] = True

            except Exception as e:
                print(f"Failed to pickle optimizer: {e}, falling back to state_dict approach")
                optimizer_state["use_state_dict"] = True
        else:
            # Restore optimizer from context state
            if optimizer_state.get("use_state_dict", False):
                print("Using state_dict approach for optimizer (not pickled)")
                # Keep current optimizer, state will be restored from checkpoints
            else:
                try:
                    # Load the pickled optimizer directly
                    optimizer = pickle.loads(optimizer_state["optimizer"])
                    optimizer_size = optimizer_state.get("size_bytes", 0)

                    # Update parameter references to current model while preserving optimizer state
                    # This is crucial because model parameters might be recreated
                    current_params = list(self.client_strategy.model.parameters())

                    # Verify parameter count matches
                    if len(optimizer.param_groups) > 0:
                        old_param_count = sum(len(group['params']) for group in optimizer.param_groups)
                        new_param_count = len(current_params)

                        if old_param_count == new_param_count:
                            # Update parameter references while preserving optimizer state and config
                            for group in optimizer.param_groups:
                                group['params'] = current_params
                            print(f"Updated {new_param_count} parameter references in optimizer")
                        else:
                            print(f"Parameter count mismatch (old: {old_param_count}, new: {new_param_count}), recreating optimizer")
                            raise ValueError("Parameter count mismatch")
                    else:
                        # Fallback: recreate param groups if empty
                        optimizer.param_groups = [{
                            'params': current_params,
                            'lr': self.config.training.learning_rate,
                            'weight_decay': self.config.training.weight_decay,
                        }]

                    self.client_strategy.optimizer = optimizer
                    print(f"Optimizer ({type(optimizer).__name__}) successfully restored from pickled state ({optimizer_size:,} bytes)")

                except Exception as e:
                    print(f"ERROR: Failed to restore pickled optimizer: {e}")
                    print("Keeping current optimizer, state will be restored from checkpoints")

    def _update_context_scheduler(self):
        """Update scheduler and optimizer state in context after training."""
        if self.context is not None:
            # Update scheduler state
            if self.client_strategy.scheduler is not None:
                scheduler_state = self.context.state.config_records["scheduler_state"]

                try:
                    # Save the scheduler object using pickle
                    scheduler_state["scheduler"] = pickle.dumps(self.client_strategy.scheduler)

                    # Update current step based on scheduler's internal state
                    if hasattr(self.client_strategy.scheduler, 'last_epoch'):
                        # For most schedulers, last_epoch represents the number of times step() has been called
                        current_step = self.client_strategy.scheduler.last_epoch
                    else:
                        # Fallback to _step_count if available
                        current_step = getattr(self.client_strategy.scheduler, '_step_count', 0)

                    scheduler_state["current_step"] = current_step
                    print(f"Updated scheduler state: current_step = {current_step}, last_epoch = {getattr(self.client_strategy.scheduler, 'last_epoch', 'N/A')}")

                except Exception as e:
                    print(f"ERROR: Failed to pickle scheduler during state update: {e}")
                    # Continue without updating the scheduler state rather than crashing

            # Update optimizer state
            if self.client_strategy.optimizer is not None:
                optimizer_state = self.context.state.config_records["optimizer_state"]

                # Only update if we're using pickle approach (not state_dict fallback)
                if not optimizer_state.get("use_state_dict", False):
                    try:
                        # Save the optimizer object using pickle
                        optimizer_pickled = pickle.dumps(self.client_strategy.optimizer)
                        optimizer_size = len(optimizer_pickled)

                        # Check size limit again (might have grown due to momentum accumulation)
                        if optimizer_size < 10 * 1024 * 1024:  # 10MB limit
                            optimizer_state["optimizer"] = optimizer_pickled
                            optimizer_state["size_bytes"] = optimizer_size
                            print(f"Updated optimizer state in context ({optimizer_size:,} bytes)")
                        else:
                            print(f"Optimizer became too large ({optimizer_size:,} bytes), switching to state_dict approach")
                            optimizer_state["use_state_dict"] = True
                            # Remove the pickled optimizer to save memory
                            if "optimizer" in optimizer_state:
                                del optimizer_state["optimizer"]

                    except Exception as e:
                        print(f"ERROR: Failed to pickle optimizer during state update: {e}")
                        print("Switching to state_dict approach for future rounds")
                        optimizer_state["use_state_dict"] = True

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

        return updated_params, len(self.train_loader.dataset), metrics

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

            return float(val_loss), len(self.val_loader.dataset), result

        except Exception as e:
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
