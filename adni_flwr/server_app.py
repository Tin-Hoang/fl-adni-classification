"""Server application for ADNI Federated Learning."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Mapping, Any
import torch
from collections.abc import Mapping

from flwr.common import Context, Metrics, Parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from adni_classification.config.config import Config
from adni_flwr.task import load_model, get_params, debug_model_architecture
from adni_flwr.strategies import StrategyFactory
from adni_flwr.server_fn import safe_weighted_average

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class FLWandbLogger:
    """WandB logger for Federated Learning."""

    def __init__(self, config: Config):
        """Initialize WandB logger."""
        self.config = config
        self.wandb_enabled = WANDB_AVAILABLE and config.wandb.use_wandb
        print(f"WandB logging: {'enabled' if self.wandb_enabled else 'disabled'}")

    def init_wandb(self):
        """Initialize WandB run."""
        if not self.wandb_enabled:
            return

        # Initialize wandb with full configuration
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.wandb.run_name,
            tags=self.config.wandb.tags,
            notes=self.config.wandb.notes,
            config=self.config.to_dict()
        )

    def log_metrics(self, metrics: Dict[str, float], prefix: str = "", step: Optional[int] = None):
        """Log metrics to WandB."""
        if not self.wandb_enabled:
            return

        try:
            # Add prefix to metric names if provided
            if prefix:
                prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            else:
                prefixed_metrics = metrics

            # Log the metrics
            if step is not None:
                wandb.log(prefixed_metrics, step=step)
            else:
                wandb.log(prefixed_metrics)
        except Exception as e:
            print(f"Error logging metrics to WandB: {e}")

    def finish(self):
        """Finish WandB run."""
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Error finishing WandB run: {e}")


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
                    # For training_time, use simple average instead of weighted average
                    if name == "training_time":
                        values = [float(value) for _, value in client_data]
                        acc_metrics[name] = sum(values) / len(values) if values else 0.0
                    else:
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

        # Debug server model after loading
        debug_model_architecture(model, "Server Model (after initialization)")

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

        # Determine which strategy to use from config - FAIL FAST if not specified
        if not hasattr(fl_config, 'strategy') or not fl_config.strategy:
            raise ValueError(
                f"ERROR: 'strategy' not specified in server config {server_config_file}. "
                f"You must explicitly set 'strategy' in the FL config section. "
                f"Available strategies: fedavg, fedprox, secagg. "
                f"This prevents dangerous implicit defaults that could cause strategy mismatch between clients and server."
            )

        strategy_name = fl_config.strategy
        print(f"Using FL strategy: {strategy_name}")

        # Validate strategy configuration
        StrategyFactory.validate_strategy_config(strategy_name, config)

        # Create strategy using factory
        try:
            # Create strategy with weighted_average function
            strategy = StrategyFactory.create_server_strategy(
                strategy_name=strategy_name,
                config=config,
                model=model,
                wandb_logger=wandb_logger,
                evaluate_metrics_aggregation_fn=weighted_average,
                fit_metrics_aggregation_fn=weighted_average
            )
        except Exception as e:
            print(f"Error creating strategy with original weighted_average: {e}")
            print("Falling back to safe_weighted_average...")

            # If that fails, try with our safe implementation
            strategy = StrategyFactory.create_server_strategy(
                strategy_name=strategy_name,
                config=config,
                model=model,
                wandb_logger=wandb_logger,
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
