"""Server application for ADNI Federated Learning."""

import os
from typing import List, Tuple, Dict, Optional
import torch
import wandb

from flwr.common import Metrics, Context, ndarrays_to_parameters, Parameters, FitRes, EvaluateRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from adni_flwr.task import load_config_from_yaml, load_model, get_params


class FLWandbLogger:
    """Wandb logger for Federated Learning metrics."""

    def __init__(self, config_path: str):
        """Initialize the Wandb logger.

        Args:
            config_path: Path to the FL server configuration file
        """
        self.config = load_config_from_yaml(config_path)
        self.initialized = False

    def init_wandb(self):
        """Initialize Wandb if enabled in the configuration."""
        if not self.initialized and self.config.get("wandb", {}).get("use_wandb", False):
            wandb_config = self.config.get("wandb", {})

            try:
                wandb.init(
                    project=wandb_config.get("project", "fl-adni-classification"),
                    entity=wandb_config.get("entity"),
                    tags=wandb_config.get("tags", ["federated-learning", "adni"]),
                    notes=wandb_config.get("notes", "Federated Learning for ADNI Classification"),
                    name=wandb_config.get("run_name", "fl-adni-run"),
                    config=self.config
                )
                self.initialized = True
                print("WandB initialized for server")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                print("Continuing without wandb logging...")

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
        *args,
        **kwargs
    ):
        """Initialize the FedADNI strategy.

        Args:
            wandb_logger: Wandb logger instance
            *args: Additional arguments for FedAvg
            **kwargs: Additional keyword arguments for FedAvg
        """
        super().__init__(*args, **kwargs)
        self.wandb_logger = wandb_logger
        self.current_round = 0

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
            for _, fit_res in results:
                client_metrics = fit_res.metrics
                if client_metrics:
                    client_id = fit_res.metrics.get("client_id", "unknown")
                    metrics_to_log = {k: v for k, v in client_metrics.items() if k != "client_id"}
                    self.wandb_logger.log_metrics(
                        metrics_to_log,
                        prefix=f"client_{client_id}",
                        step=server_round
                    )

        # Aggregate parameters (let the parent class handle this)
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Log aggregated metrics
        if self.wandb_logger and aggregated_metrics:
            self.wandb_logger.log_metrics(
                aggregated_metrics,
                prefix="server/fit",
                step=server_round
            )

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
            for _, eval_res in results:
                client_metrics = eval_res.metrics
                if client_metrics:
                    client_id = eval_res.metrics.get("client_id", "unknown")
                    metrics_to_log = {k: v for k, v in client_metrics.items() if k != "client_id"}
                    self.wandb_logger.log_metrics(
                        metrics_to_log,
                        prefix=f"client_{client_id}/",
                        step=server_round
                    )

        # Aggregate metrics (let the parent class handle this)
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Log aggregated metrics
        if self.wandb_logger:
            if aggregated_loss is not None:
                self.wandb_logger.log_metrics(
                    {"loss": aggregated_loss},
                    prefix="server/eval",
                    step=server_round
                )
            if aggregated_metrics:
                self.wandb_logger.log_metrics(
                    aggregated_metrics,
                    prefix="server/eval",
                    step=server_round
                )

        return aggregated_loss, aggregated_metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics."""
    # Filter out any metrics that are not dicts or are empty
    filtered_metrics = [(num_examples, m) for num_examples, m in metrics if isinstance(m, dict) and m]
    if not filtered_metrics:
        return {}

    # Get all metric names that are strings and present in all clients
    metric_names = [
        name for name in filtered_metrics[0][1].keys()
        if isinstance(name, str)
    ]

    acc_metrics = {}
    for name in metric_names:
        # Only aggregate if all clients have this metric and it's numeric
        if all(name in m and isinstance(m[name], (int, float)) for _, m in filtered_metrics):
            weighted_values = [m[name] * num_examples for num_examples, m in filtered_metrics]
            total_examples = sum(num_examples for num_examples, _ in filtered_metrics)
            if total_examples > 0:
                acc_metrics[name] = sum(weighted_values) / total_examples

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

    # Create WandB logger
    wandb_logger = FLWandbLogger(server_config_file)
    wandb_logger.init_wandb()

    # Load server configuration
    server_config = load_config_from_yaml(server_config_file)

    # Initialize model
    model = load_model(server_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert model parameters to flwr.common.Parameters
    ndarrays = get_params(model)
    initial_parameters = ndarrays_to_parameters(ndarrays)

    # Get FL-specific parameters from config
    fl_config = server_config.get("fl", {})
    num_rounds = fl_config.get("num_rounds", context.run_config.get("num-server-rounds", 5))
    fraction_fit = fl_config.get("fraction_fit", context.run_config.get("fraction-fit", 1.0))
    fraction_evaluate = fl_config.get("fraction_evaluate", 1.0)
    min_fit_clients = fl_config.get("min_fit_clients", 2)
    min_evaluate_clients = fl_config.get("min_evaluate_clients", 2)
    min_available_clients = fl_config.get("min_available_clients", 2)

    # Create strategy
    strategy = FedADNI(
        wandb_logger=wandb_logger,
        initial_parameters=initial_parameters,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Create server configuration
    server_config = ServerConfig(num_rounds=num_rounds)

    # Return server components
    return ServerAppComponents(strategy=strategy, config=server_config)


# Create the server app
app = ServerApp(server_fn=server_fn)
