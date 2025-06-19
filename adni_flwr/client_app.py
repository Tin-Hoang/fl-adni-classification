"""Client application for ADNI Federated Learning."""

import os
import torch
from flwr.common import Context
from flwr.client import ClientApp
from adni_classification.config.config import Config
from adni_classification.utils.training_utils import get_scheduler
from adni_flwr.task import (
    load_model,
    load_data,
    create_criterion
)
from adni_flwr.strategies import StrategyFactory, StrategyAwareClient


def client_fn(context: Context):
    """Client factory function.

    Args:
        context: Context containing client configuration

    Returns:
        An instance of NumPyClient
    """
    # Print the context
    print(f"Context: {context}")

    # Determine which GPU to use if available
    gpu_idx = context.node_config.get("gpu-id", 0)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    # Get partition ID to determine which config file to use
    partition_id = context.node_config.get("partition-id", 0)

    # Get client config files from app config
    client_config_files = context.run_config.get("client-config-files", "")
    if isinstance(client_config_files, str):
        client_config_files = [s.strip() for s in client_config_files.split(",") if s.strip()]

    # Ensure we have enough config files for all partitions
    if partition_id >= len(client_config_files):
        raise ValueError(f"Partition ID {partition_id} is out of range for {len(client_config_files)} client config files")

    # Get the specific config file for this client
    config_path = client_config_files[partition_id]
    config = Config.from_yaml(config_path)

    # Determine which strategy to use - FAIL FAST if not specified
    if not hasattr(config.fl, 'strategy') or not config.fl.strategy:
        raise ValueError(
            f"ERROR: 'strategy' not specified in client config {config_path}. "
            f"You must explicitly set 'strategy' in the FL config section. "
            f"Available strategies: fedavg, fedprox, secagg. "
            f"This prevents dangerous implicit defaults that could cause strategy mismatch between clients and server."
        )

    strategy_name = config.fl.strategy
    print(f'Initializing client {partition_id} with {strategy_name} strategy, config: {config_path} on device: {device}')

    # Use new strategy system (only path supported)
    print(f"Using new strategy system with {strategy_name} strategy")

    # Load model and create optimizer/criterion
    model = load_model(config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Get total FL rounds from run_config (set in pyproject.toml or via flwr run)
    total_fl_rounds = context.run_config.get("num-server-rounds")
    if total_fl_rounds is None:
        print("WARNING: 'num-server-rounds' not found in run_config, using default from FL config")
        total_fl_rounds = getattr(config.fl, 'num_rounds', 100)  # fallback

    print(f"Total FL rounds: {total_fl_rounds}")

    # Create learning rate scheduler with total FL rounds
    scheduler = get_scheduler(
        scheduler_type=config.training.lr_scheduler,
        optimizer=optimizer,
        num_epochs=total_fl_rounds  # Use total FL rounds for proper decay
    )

    if scheduler is not None:
        print(f"Created learning rate scheduler '{config.training.lr_scheduler}' for {total_fl_rounds} FL rounds")
    else:
        print("No learning rate scheduler specified or created")

    # Load data to create criterion
    train_loader, _ = load_data(config, batch_size=config.training.batch_size)
    criterion = create_criterion(config, train_loader.dataset, device)

    # Create client strategy with scheduler
    client_strategy = StrategyFactory.create_client_strategy(
        strategy_name=strategy_name,
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler  # Pass the properly created scheduler
    )

    # Create strategy-aware client
    client = StrategyAwareClient(config=config, device=device, client_strategy=client_strategy)

    # Attempt to resume from FL client checkpoint if specified
    if config.model.pretrained_checkpoint:
        from adni_flwr.task import is_fl_client_checkpoint
        if is_fl_client_checkpoint(config.model.pretrained_checkpoint):
            print(f"Attempting to resume strategy-aware client {partition_id} from FL checkpoint: {config.model.pretrained_checkpoint}")
            if client.resume_from_checkpoint():
                print(f"Strategy-aware client {partition_id} successfully resumed from FL checkpoint")
            else:
                print(f"Strategy-aware client {partition_id} starting fresh (FL checkpoint loading failed)")
        else:
            print(f"Strategy-aware client {partition_id}: pretrained_checkpoint is a regular model checkpoint, not resuming FL state")

    # Note: Scheduler state will be managed by server-side configure_fit() passing scheduler_step
    # This eliminates the need for checkpoint-based scheduler state restoration every round
    print(f"Client {partition_id} initialized with scheduler '{config.training.lr_scheduler}' - state will be synchronized via server")

    return client.to_client()


# Initialize the client app
app = ClientApp(client_fn=client_fn)
