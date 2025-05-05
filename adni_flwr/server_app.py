"""Server application for ADNI Federated Learning."""

import os
from typing import List, Tuple, Dict, Optional
import torch
import wandb
from collections.abc import Mapping

from flwr.common import Metrics, Context, ndarrays_to_parameters, Parameters, FitRes, EvaluateRes, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from adni_flwr.task import load_config_from_yaml, load_model, get_params
from adni_classification.config.config import Config


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

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
        # Log client evaluation metrics
        if self.wandb_logger:
            for client_proxy, eval_res in results:
                client_metrics = eval_res.metrics
                if client_metrics:
                    client_id = client_metrics.get("client_id", "unknown")
                    metrics_to_log = {k: v for k, v in client_metrics.items() if k != "client_id"}
                    self.wandb_logger.log_metrics(
                        metrics_to_log,
                        prefix=f"client_{client_id}/eval",
                        step=server_round
                    )

        # Aggregate metrics (let the parent class handle this)
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Log aggregated evaluation metrics
        if self.wandb_logger:
            if aggregated_loss is not None:
                self.wandb_logger.log_metrics(
                    {"val_aggregated_loss": aggregated_loss},
                    prefix="server",
                    step=server_round
                )
            if aggregated_metrics:
                # Remove client_id from aggregated metrics
                aggregated_metrics.pop("client_id", None)
                self.wandb_logger.log_metrics(
                    aggregated_metrics,
                    prefix="server",
                    step=server_round
                )

        # Print server model's current loss and accuracy
        if aggregated_loss is not None:
            print(f"Server model evaluation loss after round {server_round}: {aggregated_loss:.4f}")
        if aggregated_metrics:
            print(f"Server model evaluation metrics after round {server_round}:")
            for metric_name, metric_value in aggregated_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

        # Save the best model checkpoint based on the tracked metric (e.g., validation accuracy)
        if aggregated_metrics and self.metric_name in aggregated_metrics:
            self._save_best_checkpoint(self.model.state_dict(), aggregated_metrics[self.metric_name])

        return aggregated_loss, aggregated_metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics."""
    filtered_metrics = [(num_examples, dict(m)) for num_examples, m in metrics if isinstance(m, (dict, Mapping)) and m]
    if not filtered_metrics:
        return {}

    # Get all metric names that are strings and present in at least one client
    all_metric_names = set()
    for _, m in filtered_metrics:
        all_metric_names.update(name for name in m.keys() if isinstance(name, str))

    acc_metrics = {}
    for name in all_metric_names:
        # Only aggregate if all clients that reported this metric have numeric values
        client_data = [(num_examples, m.get(name)) for num_examples, m in filtered_metrics if name in m]
        if all(isinstance(value, (int, float)) for num_examples, value in client_data):
            weighted_values = [value * num_examples for num_examples, value in client_data]
            total_examples = sum(num_examples for num_examples, value in client_data)
            if total_examples > 0:
                acc_metrics[name] = sum(weighted_values) / total_examples
            elif total_examples == 0 and client_data:
                pass

    return acc_metrics


def server_fn(context: Context):
    """Server factory function.

    Args:
        context: Context containing server configuration

    Returns:
        Server components for the FL application
    """
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

    # Create strategy
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

    # Create server configuration
    server_config = ServerConfig(num_rounds=num_rounds)

    # Return server components
    return ServerAppComponents(strategy=strategy, config=server_config)


# Create the server app
app = ServerApp(server_fn=server_fn)
