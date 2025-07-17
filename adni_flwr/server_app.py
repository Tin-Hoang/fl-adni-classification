"""Server application for ADNI Federated Learning."""

import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerAppComponents, ServerConfig
from flwr.server.workflow import DefaultWorkflow

from adni_classification.config.config import Config
from adni_flwr.server_fn import safe_weighted_average
from adni_flwr.strategies import StrategyFactory
from adni_flwr.task import debug_model_architecture, get_params, load_model

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class FLWandbLogger:
    """WandB logger for Federated Learning with distributed training support."""

    def __init__(self, config: Config):
        """Initialize WandB logger."""
        self.config = config
        self.wandb_enabled = WANDB_AVAILABLE and config.wandb.use_wandb
        self.run = None
        self.run_id = None
        print(f"WandB logging: {'enabled' if self.wandb_enabled else 'disabled'}")

    def init_wandb(self, enable_shared_mode: bool = True):
        """Initialize WandB run.

        Args:
            enable_shared_mode: Whether to enable shared mode for distributed training.
                              When True, server acts as primary node and clients can join the same run.
        """
        if not self.wandb_enabled:
            return None

        try:
            # Prepare WandB settings for distributed training
            wandb_settings = None
            if enable_shared_mode:
                # Server is the primary node in distributed training
                wandb_settings = wandb.Settings(
                    mode="shared",
                    x_primary=True,
                    x_label="server",  # Unique label for server node
                )
                print("Initializing WandB in shared mode as primary node (server)")

            # Initialize wandb with full configuration
            self.run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.config.wandb.run_name,
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
                config=self.config.to_dict(),
                settings=wandb_settings,
            )

            if self.run:
                self.run_id = self.run.id
                print(f"WandB run initialized successfully. Run ID: {self.run_id}")
                if enable_shared_mode:
                    print(f"Clients can join this run using run ID: {self.run_id}")
                return self.run_id

        except Exception as e:
            print(f"Error initializing WandB: {e}")
            self.wandb_enabled = False
            return None

    def get_run_id(self) -> Optional[str]:
        """Get the current WandB run ID for sharing with clients."""
        return self.run_id

    def log_metrics(self, metrics: Dict[str, float], prefix: str = "", step: Optional[int] = None):
        """Log metrics to WandB."""
        if not self.wandb_enabled or not self.run:
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
        if self.wandb_enabled and self.run:
            try:
                wandb.finish()
                print("WandB run finished successfully")
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

        filtered_metrics = [
            (num_examples, dict(m)) for num_examples, m in metrics if isinstance(m, (dict, Mapping)) and m
        ]
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
            elif name in ["predictions_json", "labels_json", "sample_info", "client_id"] and isinstance(
                sample_value, str
            ):
                # Use the first client's value (arbitrary choice)
                acc_metrics[name] = sample_value
                print(f"Passing through string metric '{name}' with length {len(sample_value)}")

            # Other scalar values (like num_classes)
            elif name == "num_classes" and isinstance(sample_value, (int, float)):
                # Use the first client's value
                acc_metrics[name] = sample_value

            # Error for other types that shouldn't be here
            else:
                print(
                    f"Skipping metric '{name}' with type {type(sample_value).__name__} - not supported for aggregation"
                )

        print(f"Aggregated metrics keys: {list(acc_metrics.keys())}")
        return acc_metrics
    except Exception as e:
        import traceback

        print(f"Error in weighted_average: {e}")
        print(traceback.format_exc())
        return {}


def server_fn(context: Context):
    """Server factory function that auto-detects strategy and uses appropriate execution mode.

    Args:
        context: Context containing server configuration

    Returns:
        Server components for the FL application (regular strategies only)

    Note:
        For SecAgg+ strategy, this function will execute the workflow directly
        and return None to indicate the workflow has been handled.
    """
    try:
        # Get server config file from app config
        server_config_file = context.run_config.get("server-config-file")
        if not server_config_file or not os.path.exists(server_config_file):
            raise ValueError(f"Server config file not found: {server_config_file}")

        # Load the standardized Config object
        config = Config.from_yaml(server_config_file)

        # Determine which strategy to use from config - FAIL FAST if not specified
        if not hasattr(config.fl, "strategy") or not config.fl.strategy:
            raise ValueError(
                f"ERROR: 'strategy' not specified in server config {server_config_file}. "
                f"You must explicitly set 'strategy' in the FL config section. "
                f"Available strategies: fedavg, fedprox, secagg, secagg+. "
                f"This prevents dangerous implicit defaults that could cause strategy "
                f"mismatch between clients and server."
            )

        strategy_name = config.fl.strategy
        print(f"Using FL strategy: {strategy_name}")

        # AUTO-DETECTION: If SecAgg+ is detected, raise error with helpful message
        if strategy_name.lower() in ["secagg+", "secaggplus"]:
            raise ValueError(
                "🔒 SecAgg+ strategy detected!\n"
                "SecAgg+ requires workflow-based execution using @app.main() pattern.\n"
                "Your config specifies SecAgg+ but you're using the regular ServerApp pattern.\n"
                "\n"
                "Solutions:\n"
                "1. Use dedicated SecAgg+ app: flower-server-app adni_flwr.server_app:secagg_plus_app\n"
                "2. Or use auto-detecting app: flower-server-app adni_flwr.server_app:auto_app\n"
                "\n"
                "The regular 'app' cannot handle SecAgg+ due to Flower framework limitations.\n"
                "SecAgg+ requires Grid parameter that's only available in @app.main() context."
            )

        # For regular strategies (FedAvg, FedProx, SecAgg), use standard approach
        print(f"📊 Using regular strategy execution for: {strategy_name}")

        # Create WandB logger with the Config object
        wandb_logger = FLWandbLogger(config)
        enable_shared_mode = config.wandb.enable_shared_mode if hasattr(config.wandb, "enable_shared_mode") else True
        run_id = wandb_logger.init_wandb(enable_shared_mode=enable_shared_mode)

        # Store run ID for sharing with clients through FL communication
        if run_id:
            print(f"WandB run ID {run_id} will be shared with clients through FL communication")
        else:
            print("Warning: Failed to get WandB run ID for sharing with clients")

        # Initialize model using the Config object
        model = load_model(config)  # Assuming load_model can accept the Config object
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Debug server model after loading
        debug_model_architecture(model, "Server Model (after initialization)")

        # Convert initial model parameters to flwr.common.Parameters
        ndarrays = get_params(model)
        ndarrays_to_parameters(ndarrays)  # Convert to flwr parameters format

        # Get FL-specific parameters from config
        fl_config = config.fl  # Access FLConfig
        num_rounds = fl_config.num_rounds

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
                fit_metrics_aggregation_fn=weighted_average,
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
                fit_metrics_aggregation_fn=safe_weighted_average,
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


def secagg_plus_server_main(grid: Grid, context: Context):
    """Main function for SecAgg+ server that executes the workflow directly.

    This function follows the pattern from Flower's SecAgg+ examples.

    Args:
        grid: Flower Grid for workflow execution
        context: Context containing server configuration
    """
    print("🔒 SecAgg+ Server - Starting workflow-based execution")

    try:
        # Get server config file from app config
        server_config_file = context.run_config.get("server-config-file")
        if not server_config_file or not os.path.exists(server_config_file):
            raise ValueError(f"Server config file not found: {server_config_file}")

        # Load the standardized Config object
        config = Config.from_yaml(server_config_file)

        # Create WandB logger with the Config object
        wandb_logger = FLWandbLogger(config)
        enable_shared_mode = config.wandb.enable_shared_mode if hasattr(config.wandb, "enable_shared_mode") else True
        wandb_logger.init_wandb(enable_shared_mode=enable_shared_mode)

        # Initialize model using the Config object
        model = load_model(config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Debug server model after loading
        debug_model_architecture(model, "SecAgg+ Server Model (after initialization)")

        # Get FL-specific parameters from config
        fl_config = config.fl
        num_rounds = fl_config.num_rounds
        strategy_name = fl_config.strategy

        print(f"🔒 SecAgg+ Server - Using strategy: {strategy_name}")

        # Validate it's actually SecAgg+
        if strategy_name.lower() not in ["secagg+", "secaggplus"]:
            raise ValueError(f"SecAgg+ server function called with non-SecAgg+ strategy: {strategy_name}")

        # Validate strategy configuration
        StrategyFactory.validate_strategy_config(strategy_name, config)

        # Create SecAgg+ strategy
        strategy = StrategyFactory.create_server_strategy(
            strategy_name=strategy_name,
            config=config,
            model=model,
            wandb_logger=wandb_logger,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=weighted_average,
        )

        # Get the SecAgg+ workflow
        if not hasattr(strategy, "get_secagg_workflow"):
            raise ValueError("SecAgg+ strategy does not have get_secagg_workflow method")

        secagg_workflow = strategy.get_secagg_workflow()
        print(f"✅ Retrieved SecAgg+ workflow: {type(secagg_workflow).__name__}")

        # Create the main workflow with SecAgg+ fit workflow
        workflow = DefaultWorkflow(fit_workflow=secagg_workflow)

        # Create legacy context for workflow execution
        legacy_context = LegacyContext(
            context=context,
            config=ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

        # Execute the SecAgg+ workflow
        print("🚀 Executing SecAgg+ workflow...")
        workflow(grid, legacy_context)
        print("✅ SecAgg+ workflow completed successfully!")

    except Exception as e:
        print(f"❌ SecAgg+ server execution failed: {e}")
        import traceback

        traceback.print_exc()
        raise


# Create the regular server app for FedAvg, FedProx, SecAgg (NOT SecAgg+)
app = ServerApp(server_fn=server_fn)

# Create a dedicated SecAgg+ server app using the correct Flower pattern
secagg_plus_app = ServerApp()


@secagg_plus_app.main()
def main(grid: Grid, context: Context):
    """Main entry point for SecAgg+ server app using proper Flower workflow pattern."""
    print("🔒 SecAgg+ Server - Starting workflow-based execution")
    secagg_plus_server_main(grid, context)


# Create an auto-detecting server app that chooses the right execution pattern
auto_app = ServerApp()


@auto_app.main()
def auto_main(grid: Grid, context: Context):
    """Auto-detecting main entry point that chooses the right execution pattern."""
    try:
        # Get server config file from app config
        server_config_file = context.run_config.get("server-config-file")
        if not server_config_file or not os.path.exists(server_config_file):
            raise ValueError(f"Server config file not found: {server_config_file}")

        # Load the standardized Config object
        config = Config.from_yaml(server_config_file)

        # Determine which strategy to use from config
        if not hasattr(config.fl, "strategy") or not config.fl.strategy:
            raise ValueError(
                f"ERROR: 'strategy' not specified in server config {server_config_file}. "
                f"Available strategies: fedavg, fedprox, secagg, secagg+."
            )

        strategy_name = config.fl.strategy
        print(f"🔍 Auto-detected strategy: {strategy_name}")

        if strategy_name.lower() in ["secagg+", "secaggplus"]:
            print("🔒 Using SecAgg+ workflow execution")
            secagg_plus_server_main(grid, context)
        else:
            print("📊 Regular strategies require the standard ServerApp pattern")
            print("Use: flower-server-app adni_flwr.server_app:app")
            raise ValueError(
                f"Regular strategy '{strategy_name}' cannot be used with auto_app.\n"
                f"Regular strategies (fedavg, fedprox, secagg) require the standard ServerApp pattern.\n"
                f"Use: flower-server-app adni_flwr.server_app:app\n"
                f"Only SecAgg+ can be auto-detected due to its workflow requirements."
            )

    except Exception as e:
        import traceback

        print(f"❌ Auto-detection failed: {e}")
        print(traceback.format_exc())
        raise
