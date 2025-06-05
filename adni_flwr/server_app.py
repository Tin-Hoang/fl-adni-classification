"""Server application for ADNI Federated Learning."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import torch
import wandb
from collections.abc import Mapping

from flwr.common import Metrics, Context, ndarrays_to_parameters, Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, EvaluateIns
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from adni_flwr.task import load_config_from_yaml, load_model, get_params, load_data, test_with_predictions, create_criterion
from adni_classification.config.config import Config
from adni_classification.utils.visualization import plot_confusion_matrix
from adni_flwr.server_fn import safe_weighted_average


class FLWandbLogger:
    """Wandb logger for Federated Learning metrics."""

    def __init__(self, config: Config):
        """Initialize the Wandb logger.

        Args:
            config: The standardized Config object
        """
        self.config = config
        self.initialized = False

    def init_wandb(self):
        """Initialize Wandb if enabled in the configuration."""
        if not self.initialized and hasattr(self.config, 'wandb') and self.config.wandb.use_wandb:
            wandb_config = self.config.wandb
            try:
                wandb.init(
                    project=getattr(wandb_config, 'project', 'fl-adni-classification'),
                    entity=getattr(wandb_config, 'entity', None),
                    tags=getattr(wandb_config, 'tags', ['federated-learning', 'adni']),
                    notes=getattr(wandb_config, 'notes', 'Federated Learning for ADNI Classification'),
                    name=getattr(wandb_config, 'run_name', 'fl-adni-run'),
                    config=self.config.to_dict()
                )
                self.initialized = True
                print('WandB initialized for server')
            except Exception as e:
                print(f'Error initializing wandb: {e}')
                print('Continuing without wandb logging...')

    def log_metrics(self, metrics: Dict[str, float], prefix: str = "", step: Optional[int] = None):
        """Log metrics to Wandb.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix to add to metric names
            step: Current step/round number
        """
        if not self.initialized:
            return

        try:
            if prefix:
                wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            else:
                wandb_metrics = metrics

            wandb.log(wandb_metrics, step=step)
        except Exception as e:
            print(f"Error logging to wandb: {e}")

    def finish(self):
        """Finish the Wandb run."""
        if self.initialized:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Error closing wandb run: {e}")


class FedADNI(FedAvg):
    """Federated Learning strategy for ADNI classification with customized aggregation."""

    def __init__(
        self,
        wandb_logger: FLWandbLogger,
        model: torch.nn.Module,
        config: Config,
        *args,
        **kwargs
    ):
        """Initialize the FedADNI strategy.

        Args:
            wandb_logger: Wandb logger instance
            model: The PyTorch model instance used by the server
            config: The standardized Config object
            *args: Additional arguments for FedAvg
            **kwargs: Additional keyword arguments for FedAvg
        """
        super().__init__(*args, **kwargs)
        self.wandb_logger = wandb_logger
        self.model = model
        self.config = config
        self.current_round = 0
        self.checkpoint_dir = config.fl.checkpoint_dir
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
            self.criterion = create_criterion(self.config, device=self.device)

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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model updates from clients.

        Args:
            server_round: Current round number
            results: List of tuples of (client_proxy, fit_res)
            failures: List of client failures

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
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

        # Aggregate parameters and metrics (let the parent class handle this)
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Load aggregated parameters into the server model instance
        if aggregated_params is not None:
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_params)
            model_state_dict = self.model.state_dict()
            keys = list(model_state_dict.keys())
            if len(keys) == len(aggregated_ndarrays):
                new_state_dict = {keys[i]: torch.tensor(aggregated_ndarrays[i]) for i in range(len(keys))}
                self.model.load_state_dict(new_state_dict)
                print(f"Server model updated with aggregated parameters for round {server_round}")
            else:
                print(f"Warning: Number of aggregated parameters ({len(aggregated_ndarrays)}) does not match model state_dict keys ({len(keys)}). Cannot load parameters.")

        # Log aggregated fit metrics
        if self.wandb_logger and aggregated_metrics:
            # Remove client_id from aggregated metrics
            aggregated_metrics.pop("client_id", None)
            self.wandb_logger.log_metrics(
                aggregated_metrics,
                prefix="server",
                step=server_round
            )
        # Print server model's current loss and accuracy
        if aggregated_metrics:
            print(f"Server model metrics after round {server_round}:")
            for metric_name, metric_value in aggregated_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

        # Save frequency checkpoint
        if server_round % self.config.training.checkpoint.save_frequency == 0:
            self._save_checkpoint(self.model.state_dict(), server_round)

        return aggregated_params, aggregated_metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        Args:
            server_round: Current round number
            parameters: Current global model parameters
            client_manager: Client manager

        Returns:
            List of tuples (client_proxy, evaluate_ins)
        """
        # Get the base configuration from the parent class
        client_instructions = super().configure_evaluate(
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
        """Aggregate evaluation results from clients.

        Args:
            server_round: Current round number
            results: List of tuples of (client_proxy, evaluate_res)
            failures: List of client failures

        Returns:
            Tuple of (aggregated_loss, metrics)
        """
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
                            "global_loss": server_val_loss
                        },
                        prefix="server",
                        step=server_round
                    )

                # Save best checkpoint based on server validation
                self._save_best_checkpoint(self.model.state_dict(), server_val_accuracy)

            return None, {
                "no_client_evaluation": True,
                "server_val_accuracy": server_val_accuracy or 0.0,
                "server_evaluation_skipped": not should_evaluate_server
            }

        try:
            # Log client evaluation metrics for clients that actually evaluated
            if self.wandb_logger:
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
                                    client_title = f'Confusion Matrix - Client {client_id} - Round {server_round}'
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
                    global_title = f'Global Confusion Matrix (Server Evaluation) - Round {server_round}'
                    global_fig = plot_confusion_matrix(
                        y_true=np.array(server_labels),
                        y_pred=np.array(server_predictions),
                        class_names=class_names,
                        normalize=False,
                        title=global_title
                    )

                    # Log to wandb
                    if self.wandb_logger:
                        self.wandb_logger.log_metrics(
                            {
                                "global_confusion_matrix": wandb.Image(global_fig),
                                "global_accuracy": server_val_accuracy,
                                "global_loss": server_val_loss
                            },
                            prefix="server",
                            step=server_round
                        )

                    plt.close(global_fig)
                    print(f"Logged global confusion matrix from server evaluation - Accuracy: {server_val_accuracy:.2f}%")

                except Exception as e:
                    print(f"Error creating global confusion matrix from server evaluation: {e}")

            # Aggregate metrics from clients that actually evaluated (let the parent class handle this)
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
                server_round, actual_results, failures
            )

            print(f"Aggregated loss: {aggregated_loss}, metrics keys: {aggregated_metrics.keys() if aggregated_metrics else 'None'}")

            # Log aggregated evaluation metrics and server-side metrics
            if self.wandb_logger:
                try:
                    if aggregated_loss is not None:
                        self.wandb_logger.log_metrics(
                            {"val_aggregated_loss": aggregated_loss},
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
                                "global_loss": server_val_loss
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
                print(f"Server model evaluation metrics after round {server_round}:")
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
            if should_evaluate_server and server_val_accuracy is not None:
                aggregated_metrics["server_val_accuracy"] = server_val_accuracy

            return aggregated_loss, aggregated_metrics

        except Exception as e:
            import traceback
            print(f"Error in aggregate_evaluate: {e}")
            print(traceback.format_exc())
            return None, {"server_evaluation_performed": should_evaluate_server}


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics.

    Only processes scalar metrics (int, float) and passes through string metrics.
    JSON-encoded lists will be passed through for later decoding.
    """
    try:
        if not metrics:
            print("WARNING: weighted_average received empty metrics list")
            return {}

        print(f"Weighted average received {len(metrics)} metrics")

        filtered_metrics = [(num_examples, dict(m)) for num_examples, m in metrics if isinstance(m, (dict, Mapping)) and m]
        if not filtered_metrics:
            print("WARNING: weighted_average filtered metrics list is empty after filtering")
            return {}

        # Get all metric names that are present in at least one client
        all_metric_names = set()
        for _, m in filtered_metrics:
            all_metric_names.update(name for name in m.keys() if isinstance(name, str))

        print(f"Metric names present: {all_metric_names}")

        acc_metrics = {}
        for name in all_metric_names:
            # Get all clients that reported this metric
            client_data = [(num_examples, m.get(name)) for num_examples, m in filtered_metrics if name in m]

            # Skip empty data
            if not client_data:
                continue

            # Sample value to determine type
            sample_value = client_data[0][1]

            # Handle scalar metrics (compute weighted average)
            if all(isinstance(value, (int, float)) for _, value in client_data):
                try:
                    weighted_values = [float(value) * num_examples for num_examples, value in client_data]
                    total_examples = sum(num_examples for num_examples, _ in client_data)
                    if total_examples > 0:
                        acc_metrics[name] = sum(weighted_values) / total_examples
                except Exception as e:
                    print(f"Error processing scalar metric '{name}': {e}")

            # Pass through string metrics (like JSON-encoded lists)
            elif name in ["predictions_json", "labels_json", "sample_info", "client_id"] and isinstance(sample_value, str):
                # Use the first client's value (arbitrary choice)
                acc_metrics[name] = sample_value
                print(f"Passing through string metric '{name}' with length {len(sample_value)}")

            # Other scalar values (like num_classes)
            elif name == "num_classes" and isinstance(sample_value, (int, float)):
                # Use the first client's value
                acc_metrics[name] = sample_value

            # Error for other types that shouldn't be here
            else:
                print(f"Skipping metric '{name}' with type {type(sample_value).__name__} - not supported for aggregation")

        print(f"Aggregated metrics keys: {list(acc_metrics.keys())}")
        return acc_metrics
    except Exception as e:
        import traceback
        print(f"Error in weighted_average: {e}")
        print(traceback.format_exc())
        return {}


def server_fn(context: Context):
    """Server factory function.

    Args:
        context: Context containing server configuration

    Returns:
        Server components for the FL application
    """
    try:
        # Get server config file from app config
        server_config_file = context.run_config.get("server-config-file")
        if not server_config_file or not os.path.exists(server_config_file):
            raise ValueError(f"Server config file not found: {server_config_file}")

        # Load the standardized Config object
        config = Config.from_yaml(server_config_file)

        # Create WandB logger with the Config object
        wandb_logger = FLWandbLogger(config)
        wandb_logger.init_wandb()

        # Initialize model using the Config object
        model = load_model(config)  # Assuming load_model can accept the Config object
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Convert initial model parameters to flwr.common.Parameters
        ndarrays = get_params(model)
        initial_parameters = ndarrays_to_parameters(ndarrays)

        # Get FL-specific parameters from config
        fl_config = config.fl  # Access FLConfig
        num_rounds = fl_config.num_rounds
        fraction_fit = fl_config.fraction_fit
        fraction_evaluate = fl_config.fraction_evaluate
        min_fit_clients = fl_config.min_fit_clients
        min_evaluate_clients = fl_config.min_evaluate_clients
        min_available_clients = fl_config.min_available_clients

        # Use a try-except block to handle potential serialization issues
        try:
            # First try with the original weighted_average function
            strategy = FedADNI(
                wandb_logger=wandb_logger,
                model=model,
                config=config,
                initial_parameters=initial_parameters,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_metrics_aggregation_fn=weighted_average,
                fit_metrics_aggregation_fn=weighted_average
            )
        except Exception as e:
            print(f"Error creating strategy with original weighted_average: {e}")
            print("Falling back to safe_weighted_average...")

            # If that fails, try with our safe implementation
            strategy = FedADNI(
                wandb_logger=wandb_logger,
                model=model,
                config=config,
                initial_parameters=initial_parameters,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_metrics_aggregation_fn=safe_weighted_average,
                fit_metrics_aggregation_fn=safe_weighted_average
            )

        # Create server configuration
        server_config = ServerConfig(num_rounds=num_rounds)

        # Return server components
        return ServerAppComponents(strategy=strategy, config=server_config)

    except Exception as e:
        import traceback
        print(f"Error in server_fn: {e}")
        print(traceback.format_exc())
        # Still need to return a ServerAppComponents object
        raise


# Create the server app
app = ServerApp(server_fn=server_fn)
