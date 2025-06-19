"""Secure Aggregation (SecAgg) strategy implementation."""

import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Union
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


class SecAggStrategy(FLStrategyBase):
    """Server-side Secure Aggregation strategy with comprehensive WandB logging."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        wandb_logger: Optional[Any] = None,
        noise_multiplier: float = 0.1,
        dropout_rate: float = 0.0,
        **kwargs
    ):
        """Initialize SecAgg strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            wandb_logger: Wandb logger instance
            noise_multiplier: Multiplier for noise addition
            dropout_rate: Dropout rate for parameter masking
            **kwargs: Additional SecAgg parameters
        """
        super().__init__(config, model, wandb_logger, **kwargs)

        self.noise_multiplier = noise_multiplier
        self.dropout_rate = dropout_rate
        self.client_masks = {}  # Store client masks for secure aggregation

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
        self.checkpoint_dir = config.checkpoint_dir
        self.best_metric = None
        self.metric_name = "val_accuracy"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Load server-side validation dataset for global evaluation
        self._load_server_validation_data()

    def _load_server_validation_data(self):
        """Load validation dataset on the server for global evaluation."""
        try:
            print("Loading server-side validation dataset...")
            # Load only the validation loader for server evaluation
            _, self.server_val_loader = load_data(
                config=self.config,
                batch_size=self.config.training.batch_size
            )

            # Create criterion for evaluation
            self.criterion = create_criterion(self.config, train_dataset=None, device=self.device)

            print(f"Server validation dataset loaded with {len(self.server_val_loader)} batches")
        except Exception as e:
            print(f"Warning: Could not load server validation data: {e}")
            print("Global accuracy evaluation will be skipped.")
            self.server_val_loader = None
            self.criterion = None

    def _evaluate_server_model(self, server_round: int) -> Tuple[Optional[float], Optional[float], Optional[List], Optional[List]]:
        """Evaluate the server model on the validation dataset.

        Args:
            server_round: Current round number

        Returns:
            Tuple of (loss, accuracy, predictions, labels) or (None, None, None, None) if evaluation fails
        """
        if self.server_val_loader is None or self.criterion is None:
            return None, None, None, None

        try:
            print(f"Evaluating server model on validation set for round {server_round}...")

            # Evaluate the server model
            val_loss, val_accuracy, predictions, labels = test_with_predictions(
                model=self.model,
                test_loader=self.server_val_loader,
                criterion=self.criterion,
                device=self.device,
                mixed_precision=self.config.training.mixed_precision
            )

            print(f"Server validation results - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
            return val_loss, val_accuracy, predictions, labels

        except Exception as e:
            print(f"Error evaluating server model: {e}")
            return None, None, None, None

    def _save_checkpoint(self, model_state_dict: dict, round_num: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"round_{round_num}.pt")
        torch.save(model_state_dict, checkpoint_path)
        print(f"Saved checkpoint for round {round_num} to {checkpoint_path}")

    def _save_best_checkpoint(self, model_state_dict: dict, metric: float):
        """Save the best model checkpoint based on the given metric."""
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric
            best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(model_state_dict, best_checkpoint_path)
            print(f"Saved new best model checkpoint with {self.metric_name} {metric:.4f} to {best_checkpoint_path}")

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "secagg"

    def get_strategy_params(self) -> Dict[str, Any]:
        """Return strategy-specific parameters."""
        return {
            "noise_multiplier": self.noise_multiplier,
            "dropout_rate": self.dropout_rate,
            "fraction_fit": self.fedavg_strategy.fraction_fit,
            "fraction_evaluate": self.fedavg_strategy.fraction_evaluate,
            "min_fit_clients": self.fedavg_strategy.min_fit_clients,
            "min_evaluate_clients": self.fedavg_strategy.min_evaluate_clients,
            "min_available_clients": self.fedavg_strategy.min_available_clients,
        }

    def generate_client_mask(self, client_id: str, round_num: int) -> np.ndarray:
        """Generate a deterministic mask for a client.

        Args:
            client_id: Client identifier
            round_num: Current round number

        Returns:
            Numpy array mask
        """
        # Create deterministic seed from client_id and round
        seed_str = f"{client_id}_{round_num}_{self.noise_multiplier}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)

        # Set numpy random seed for reproducibility
        np.random.seed(seed)

        # Generate mask with same shape as model parameters
        model_params = get_params(self.model)
        mask = []

        for param_array in model_params:
            # Generate random mask for each parameter array
            param_mask = np.random.uniform(-1, 1, param_array.shape)
            mask.append(param_mask)

        return mask

    def secure_aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        round_num: int
    ) -> Optional[Parameters]:
        """Perform secure aggregation of client updates.

        Args:
            results: List of client results
            round_num: Current round number

        Returns:
            Aggregated parameters
        """
        if not results:
            return None

        print(f"Performing secure aggregation for {len(results)} clients")

        # Collect masked parameters and weights
        masked_params_list = []
        weights = []
        client_masks = []

        for client_proxy, fit_res in results:
            client_id = str(client_proxy.cid)

            # Generate mask for this client
            client_mask = self.generate_client_mask(client_id, round_num)
            client_masks.append(client_mask)

            # Get client parameters
            client_params = parameters_to_ndarrays(fit_res.parameters)

            # Apply mask to parameters (remove the mask that was added by client)
            unmasked_params = []
            for param, mask in zip(client_params, client_mask):
                unmasked_param = param - mask
                unmasked_params.append(unmasked_param)

            masked_params_list.append(unmasked_params)
            weights.append(fit_res.num_examples)

        # Perform weighted average
        total_examples = sum(weights)
        if total_examples == 0:
            return None

        # Initialize aggregated parameters
        aggregated_params = []

        for i in range(len(masked_params_list[0])):
            # Weighted sum for each parameter
            weighted_sum = np.zeros_like(masked_params_list[0][i])

            for j, (params, weight) in enumerate(zip(masked_params_list, weights)):
                weighted_sum += params[i] * (weight / total_examples)

            aggregated_params.append(weighted_sum)

        print(f"Secure aggregation completed with {len(results)} clients")

        return ndarrays_to_parameters(aggregated_params)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate fit results using secure aggregation with enhanced logging."""
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

        # Perform secure aggregation
        aggregated_parameters = self.secure_aggregate(results, server_round)

        if aggregated_parameters is None:
            return None, {}

        # Update server model with aggregated parameters
        param_arrays = safe_parameters_to_ndarrays(aggregated_parameters)
        set_params(self.model, param_arrays)

        # Collect metrics with SecAgg-specific information
        metrics = {
            "num_clients": len(results),
            "num_failures": len(failures),
            "noise_multiplier": self.noise_multiplier,
            "dropout_rate": self.dropout_rate,
        }

        # Log aggregated fit metrics with SecAgg-specific information
        if self.wandb_logger and metrics:
            metrics_with_secagg = metrics.copy()
            self.wandb_logger.log_metrics(
                metrics_with_secagg,
                prefix="server",
                step=server_round
            )

        # Print server model's current metrics
        print(f"Server model metrics after round {server_round} (SecAgg noise={self.noise_multiplier}, dropout={self.dropout_rate}):")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")

        # Save frequency checkpoint
        if server_round % self.config.training.checkpoint.save_frequency == 0:
            self._save_checkpoint(self.model.state_dict(), server_round)

        return aggregated_parameters, metrics

    # Implement required Strategy abstract methods
    def initialize_parameters(self, client_manager):
        """Initialize global model parameters."""
        # Instead of delegating to fedavg_strategy (which returns None),
        # provide initial parameters from our server model
        from adni_flwr.task import get_params
        from flwr.common import ndarrays_to_parameters

        print("SecAggStrategy: Initializing parameters from server model")
        ndarrays = get_params(self.model)
        print(f"SecAggStrategy: Sending {len(ndarrays)} parameter arrays to clients")
        print(f"SecAggStrategy: First few parameter shapes: {[arr.shape for arr in ndarrays[:5]]}")

        return ndarrays_to_parameters(ndarrays)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get the base configuration from the parent class
        client_instructions = self.fedavg_strategy.configure_fit(
            server_round, parameters, client_manager
        )

        # Add server_round, scheduler information, and SecAgg parameters to the config for each client
        updated_instructions = []
        for client_proxy, fit_ins in client_instructions:
            # Add server_round, scheduler step info, and SecAgg parameters to the existing config
            config = fit_ins.config.copy() if fit_ins.config else {}
            config["server_round"] = server_round
            # For scheduler continuity: send the step count (server_round - 1 since we start from 0)
            config["scheduler_step"] = server_round - 1
            config["noise_multiplier"] = self.noise_multiplier
            config["dropout_rate"] = self.dropout_rate

            # Create new FitIns with updated config
            updated_fit_ins = FitIns(
                parameters=fit_ins.parameters,
                config=config
            )
            updated_instructions.append((client_proxy, updated_fit_ins))

        return updated_instructions

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

        return updated_instructions

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results with enhanced logging."""
        # Print information about results and failures
        print(f"aggregate_evaluate: received {len(results)} results and {len(failures)} failures")

        # Check if server should evaluate in this round based on frequency
        evaluate_frequency = getattr(self.config.fl, 'evaluate_frequency', 1)
        should_evaluate_server = server_round % evaluate_frequency == 0

        # Count clients that skipped evaluation vs those that actually evaluated
        skipped_count = 0
        evaluated_count = 0
        actual_results = []

        for client_proxy, eval_res in results:
            client_metrics = eval_res.metrics
            if client_metrics and client_metrics.get("evaluation_skipped", False):
                skipped_count += 1
                print(f"Client {client_metrics.get('client_id', 'unknown')} skipped evaluation for round {server_round}")
            else:
                evaluated_count += 1
                actual_results.append((client_proxy, eval_res))

        print(f"Round {server_round}: {evaluated_count} clients evaluated, {skipped_count} clients skipped evaluation")
        print(f"Server evaluation: {'enabled' if should_evaluate_server else 'skipped'} for round {server_round} (evaluating every {evaluate_frequency} rounds)")

        # Print details about any failures
        if failures:
            for i, failure in enumerate(failures):
                print(f"Failure {i+1}: {type(failure).__name__}: {str(failure)}")

        # Initialize server evaluation variables
        server_val_loss, server_val_accuracy, server_predictions, server_labels = None, None, None, None

        # Evaluate server model only if it's the right frequency
        if should_evaluate_server:
            server_val_loss, server_val_accuracy, server_predictions, server_labels = self._evaluate_server_model(server_round)
        else:
            print(f"Skipping server-side evaluation for round {server_round}")

        # If no clients actually evaluated, handle gracefully
        if not actual_results:
            print("WARNING: No clients performed evaluation in this round")

            if should_evaluate_server and server_val_loss is not None and server_val_accuracy is not None:
                print(f"Server validation results for round {server_round}:")
                print(f"  Server validation loss: {server_val_loss:.4f}")
                print(f"  Server validation accuracy: {server_val_accuracy:.2f}%")

                # Log server metrics even if no clients evaluated
                if self.wandb_logger:
                    self.wandb_logger.log_metrics(
                        {
                            "global_accuracy": server_val_accuracy,
                            "global_loss": server_val_loss,
                            "noise_multiplier": self.noise_multiplier,
                            "dropout_rate": self.dropout_rate
                        },
                        prefix="server",
                        step=server_round
                    )

                # Save best checkpoint based on server validation
                self._save_best_checkpoint(self.model.state_dict(), server_val_accuracy)

            return None, {
                "no_client_evaluation": True,
                "server_val_accuracy": server_val_accuracy or 0.0,
                "server_evaluation_skipped": not should_evaluate_server,
                "noise_multiplier": self.noise_multiplier,
                "dropout_rate": self.dropout_rate
            }

        try:
            # Log client evaluation metrics for clients that actually evaluated
            if self.wandb_logger and WANDB_AVAILABLE:
                for client_proxy, eval_res in actual_results:
                    try:
                        client_metrics = eval_res.metrics
                        if not client_metrics:
                            print(f"WARNING: Client metrics are empty for a client")
                            continue

                        client_id = client_metrics.get("client_id", "unknown")
                        print(f"Processing metrics from client {client_id}")

                        # Debug: print all keys in client metrics
                        print(f"Client {client_id} metrics keys: {list(client_metrics.keys())}")

                        # Extract and decode JSON predictions and labels if present for client-specific confusion matrices
                        if "predictions_json" in client_metrics and "labels_json" in client_metrics:
                            try:
                                import json

                                # Decode JSON strings to lists
                                predictions_json = client_metrics.get("predictions_json", "[]")
                                labels_json = client_metrics.get("labels_json", "[]")
                                sample_info = client_metrics.get("sample_info", "unknown")

                                # Parse the JSON strings
                                predictions = json.loads(predictions_json)
                                labels = json.loads(labels_json)

                                print(f"Client {client_id}: Decoded {len(predictions)} predictions and {len(labels)} labels. Sample info: {sample_info}")

                                # Get the number of classes
                                num_classes = client_metrics.get("num_classes", 3)

                                # Generate confusion matrix for this client
                                if len(predictions) > 0 and len(labels) > 0:
                                    # Set class names based on classification mode
                                    class_names = ["CN", "AD"] if num_classes == 2 else ["CN", "MCI", "AD"]

                                    # Create the confusion matrix
                                    from sklearn.metrics import confusion_matrix
                                    client_cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

                                    # Create a figure for the confusion matrix
                                    client_title = f'Confusion Matrix - Client {client_id} - Round {server_round} (SecAgg)'
                                    if sample_info != "full_dataset":
                                        client_title += f" ({sample_info})"

                                    # Plot using the original visualization function
                                    client_fig = plot_confusion_matrix(
                                        y_true=np.array(labels),
                                        y_pred=np.array(predictions),
                                        class_names=class_names,
                                        normalize=False,
                                        title=client_title
                                    )

                                    # Log to wandb
                                    self.wandb_logger.log_metrics(
                                        {"confusion_matrix": wandb.Image(client_fig)},
                                        prefix=f"client_{client_id}/eval",
                                        step=server_round
                                    )
                                    plt.close(client_fig)
                            except Exception as e:
                                print(f"Error decoding predictions/labels from client {client_id}: {e}")

                        # Log other scalar metrics (excluding the encoded data and metadata)
                        try:
                            metrics_to_log = {
                                k: v for k, v in client_metrics.items()
                                if k not in ["predictions_json", "labels_json", "sample_info", "client_id", "error", "num_classes", "evaluation_skipped", "evaluation_frequency", "current_round"]
                                and isinstance(v, (int, float))
                            }

                            self.wandb_logger.log_metrics(
                                metrics_to_log,
                                prefix=f"client_{client_id}/eval",
                                step=server_round
                            )
                        except Exception as e:
                            print(f"Error logging metrics for client {client_id}: {e}")
                    except Exception as e:
                        print(f"Error processing evaluation result: {e}")

            # Create and log global confusion matrix using server evaluation (only if server evaluated)
            if should_evaluate_server and server_predictions is not None and server_labels is not None:
                try:
                    # Determine the number of classes
                    num_classes = 2 if self.config.data.classification_mode == "CN_AD" else 3
                    class_names = ["CN", "AD"] if num_classes == 2 else ["CN", "MCI", "AD"]

                    print(f"Creating global confusion matrix from server evaluation with {len(server_predictions)} predictions")

                    # Create the confusion matrix
                    from sklearn.metrics import confusion_matrix
                    global_cm = confusion_matrix(server_labels, server_predictions, labels=list(range(num_classes)))

                    # Plot the global confusion matrix
                    global_title = f'Global Confusion Matrix (Server Evaluation) - Round {server_round} (SecAgg)'
                    global_fig = plot_confusion_matrix(
                        y_true=np.array(server_labels),
                        y_pred=np.array(server_predictions),
                        class_names=class_names,
                        normalize=False,
                        title=global_title
                    )

                    # Log to wandb
                    if self.wandb_logger and WANDB_AVAILABLE:
                        self.wandb_logger.log_metrics(
                            {
                                "global_confusion_matrix": wandb.Image(global_fig),
                                "global_accuracy": server_val_accuracy,
                                "global_loss": server_val_loss,
                                "noise_multiplier": self.noise_multiplier,
                                "dropout_rate": self.dropout_rate
                            },
                            prefix="server",
                            step=server_round
                        )

                    plt.close(global_fig)
                    print(f"Logged global confusion matrix from server evaluation - Accuracy: {server_val_accuracy:.2f}%")

                except Exception as e:
                    print(f"Error creating global confusion matrix from server evaluation: {e}")

            # Aggregate metrics from clients that actually evaluated using base FedAvg
            aggregated_loss, aggregated_metrics = self.fedavg_strategy.aggregate_evaluate(
                server_round, actual_results, failures
            )

            print(f"Aggregated loss: {aggregated_loss}, metrics keys: {aggregated_metrics.keys() if aggregated_metrics else 'None'}")

            # Log aggregated evaluation metrics and server-side metrics
            if self.wandb_logger:
                try:
                    if aggregated_loss is not None:
                        self.wandb_logger.log_metrics(
                            {
                                "val_aggregated_loss": aggregated_loss,
                                "noise_multiplier": self.noise_multiplier,
                                "dropout_rate": self.dropout_rate
                            },
                            prefix="server",
                            step=server_round
                        )
                    if aggregated_metrics:
                        # Filter out non-scalar and special keys
                        filtered_metrics = {
                            k: v for k, v in aggregated_metrics.items()
                            if k not in ["predictions_json", "labels_json", "sample_info", "client_id", "error", "num_classes", "evaluation_skipped", "evaluation_frequency", "current_round"]
                            and isinstance(v, (int, float))
                        }
                        # Add SecAgg-specific information
                        filtered_metrics["noise_multiplier"] = self.noise_multiplier
                        filtered_metrics["dropout_rate"] = self.dropout_rate

                        if filtered_metrics:  # Only log if there are metrics left
                            self.wandb_logger.log_metrics(
                                filtered_metrics,
                                prefix="server",
                                step=server_round
                            )

                    # Log server-side metrics only if server evaluated
                    if should_evaluate_server and server_val_loss is not None and server_val_accuracy is not None:
                        self.wandb_logger.log_metrics(
                            {
                                "global_accuracy": server_val_accuracy,
                                "global_loss": server_val_loss,
                                "noise_multiplier": self.noise_multiplier,
                                "dropout_rate": self.dropout_rate
                            },
                            prefix="server",
                            step=server_round
                        )
                except Exception as e:
                    print(f"Error logging aggregated metrics: {e}")

            # Print server model's current loss and accuracy
            if aggregated_loss is not None:
                print(f"Server model evaluation loss after round {server_round}: {aggregated_loss:.4f}")
            if aggregated_metrics:
                print(f"Server model evaluation metrics after round {server_round} (SecAgg noise={self.noise_multiplier}, dropout={self.dropout_rate}):")
                for metric_name, metric_value in aggregated_metrics.items():
                    if metric_name not in ["predictions_json", "labels_json", "sample_info", "client_id", "error", "num_classes", "evaluation_skipped", "evaluation_frequency", "current_round"] and isinstance(metric_value, (int, float)):
                        print(f"  {metric_name}: {metric_value:.4f}")

            # Print server-side validation results only if server evaluated
            if should_evaluate_server and server_val_loss is not None and server_val_accuracy is not None:
                print(f"Server validation results after round {server_round}:")
                print(f"  Server validation loss: {server_val_loss:.4f}")
                print(f"  Server validation accuracy: {server_val_accuracy:.2f}%")

                # Save the best model checkpoint based on server validation accuracy
                try:
                    self._save_best_checkpoint(self.model.state_dict(), server_val_accuracy)
                except Exception as e:
                    print(f"Error saving best checkpoint: {e}")
            elif should_evaluate_server:
                print(f"Server evaluation was attempted but failed for round {server_round}")

            # Fallback to aggregated metrics if server evaluation is not available or not performed
            if not should_evaluate_server or server_val_accuracy is None:
                if aggregated_metrics and self.metric_name in aggregated_metrics:
                    try:
                        metric_value = aggregated_metrics[self.metric_name]
                        if isinstance(metric_value, (int, float)):
                            self._save_best_checkpoint(self.model.state_dict(), metric_value)
                        else:
                            print(f"Cannot save checkpoint: metric {self.metric_name} is not a number")
                    except Exception as e:
                        print(f"Error saving best checkpoint: {e}")

            # Add server evaluation info to aggregated metrics
            if aggregated_metrics is None:
                aggregated_metrics = {}

            aggregated_metrics["server_evaluation_performed"] = should_evaluate_server
            aggregated_metrics["noise_multiplier"] = self.noise_multiplier
            aggregated_metrics["dropout_rate"] = self.dropout_rate
            if should_evaluate_server and server_val_accuracy is not None:
                aggregated_metrics["server_val_accuracy"] = server_val_accuracy

            return aggregated_loss, aggregated_metrics

        except Exception as e:
            import traceback
            print(f"Error in aggregate_evaluate: {e}")
            print(traceback.format_exc())
            return None, {
                "server_evaluation_performed": should_evaluate_server,
                "noise_multiplier": self.noise_multiplier,
                "dropout_rate": self.dropout_rate
            }

    def evaluate(self, server_round, parameters):
        """Evaluate model parameters."""
        return self.fedavg_strategy.evaluate(server_round, parameters)

    # Delegate other attributes to FedAvg
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying FedAvg strategy."""
        return getattr(self.fedavg_strategy, name)


class SecAggClient(ClientStrategyBase):
    """Client-side Secure Aggregation strategy."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        noise_multiplier: float = 0.1,
        dropout_rate: float = 0.0,
        **kwargs
    ):
        """Initialize SecAgg client strategy.

        Args:
            config: Configuration object
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to use for computation
            scheduler: Learning rate scheduler (optional)
            noise_multiplier: Multiplier for noise addition
            dropout_rate: Dropout rate for parameter masking
            **kwargs: Additional strategy parameters
        """
        super().__init__(config, model, optimizer, criterion, device, scheduler, **kwargs)

        self.noise_multiplier = noise_multiplier
        self.dropout_rate = dropout_rate
        # Client ID must be explicitly set - FAIL FAST if not specified
        if not hasattr(config.fl, 'client_id') or config.fl.client_id is None:
            raise ValueError(
                "ERROR: 'client_id' not specified in client config. "
                "You must explicitly set 'client_id' in the FL config section. "
                "This prevents client identification issues in federated learning."
            )
        self.client_id = config.fl.client_id
        self.current_round = 0

        # SecAgg-specific parameters
        self.mixed_precision = config.training.mixed_precision
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        return "secagg"

    def generate_client_mask(self, round_num: int) -> List[np.ndarray]:
        """Generate a deterministic mask for this client.

        Args:
            round_num: Current round number

        Returns:
            List of numpy array masks
        """
        # Create deterministic seed from client_id and round
        seed_str = f"{self.client_id}_{round_num}_{self.noise_multiplier}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)

        # Set numpy random seed for reproducibility
        np.random.seed(seed)

        # Generate mask with same shape as model parameters
        model_params = get_params(self.model)
        mask = []

        for param_array in model_params:
            # Generate random mask for each parameter array
            param_mask = np.random.uniform(-1, 1, param_array.shape)
            mask.append(param_mask)

        return mask

    def add_noise_to_parameters(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """Add noise to parameters for privacy.

        Args:
            params: List of parameter arrays

        Returns:
            List of noisy parameter arrays
        """
        noisy_params = []

        for param_array in params:
            # Add Gaussian noise
            noise = np.random.normal(0, self.noise_multiplier, param_array.shape)
            noisy_param = param_array + noise
            noisy_params.append(noisy_param)

        return noisy_params

    def apply_dropout_mask(self, params: List[np.ndarray]) -> List[np.ndarray]:
        """Apply dropout mask to parameters.

        Args:
            params: List of parameter arrays

        Returns:
            List of masked parameter arrays
        """
        if self.dropout_rate == 0.0:
            return params

        masked_params = []

        for param_array in params:
            # Create dropout mask
            mask = np.random.binomial(1, 1-self.dropout_rate, param_array.shape)
            masked_param = param_array * mask
            masked_params.append(masked_param)

        return masked_params

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

        # Update scheduler state based on server-provided step information
        scheduler_step = round_config.get("scheduler_step", 0)
        if self.scheduler is not None and scheduler_step > 0:
            # Fast-forward the scheduler to the correct step
            current_lr_before = self.optimizer.param_groups[0]['lr']
            for _ in range(scheduler_step):
                self.scheduler.step()
            current_lr_after = self.optimizer.param_groups[0]['lr']
            print(f"SecAggClient: Scheduler fast-forwarded to step {scheduler_step}, LR: {current_lr_before:.8f} -> {current_lr_after:.8f}")

        # Reset optimizer state
        self.optimizer.zero_grad()

        # Update parameters if specified in round config
        if "noise_multiplier" in round_config:
            self.noise_multiplier = round_config["noise_multiplier"]
        if "dropout_rate" in round_config:
            self.dropout_rate = round_config["dropout_rate"]

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        **kwargs
    ) -> Tuple[float, float]:
        """Train the model for one epoch using SecAgg.

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

    def get_secure_parameters(self) -> List[np.ndarray]:
        """Get model parameters with secure masking applied.

        Returns:
            List of masked parameter arrays
        """
        # Get current model parameters
        params = get_params(self.model)

        # Apply dropout mask
        params = self.apply_dropout_mask(params)

        # Add noise for privacy
        params = self.add_noise_to_parameters(params)

        # Generate and apply client mask
        client_mask = self.generate_client_mask(self.current_round)
        masked_params = []

        for param, mask in zip(params, client_mask):
            masked_param = param + mask
            masked_params.append(masked_param)

        return masked_params

    def get_custom_metrics(self) -> Dict[str, Any]:
        """Return custom SecAgg-specific metrics.

        Returns:
            Dictionary of custom metrics
        """
        # Return empty dict to avoid logging non-essential configuration metrics to WandB
        return {}
