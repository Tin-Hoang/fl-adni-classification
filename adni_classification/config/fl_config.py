from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FLConfig:
    """Federated Learning configuration."""
    num_rounds: int = 10
    strategy: str = "fedavg"  # fedavg, fedprox, secagg
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    local_epochs: int = 1
    client_config_files: List[str] = None
    # checkpoint_dir: str = "checkpoints"  # Removed: using training.output_dir instead
    evaluate_frequency: int = 1  # Run evaluation every N rounds (1 means every round)
    # FedProx specific parameters
    fedprox_mu: float = 0.01
    # SecAgg specific parameters
    secagg_noise_multiplier: float = 1.0
    secagg_dropout_rate: float = 0.0
    # Client ID (used for client applications)
    client_id: Optional[int] = None
