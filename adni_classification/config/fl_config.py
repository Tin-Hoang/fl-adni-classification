from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class FLConfig:
    """Federated Learning configuration."""
    # Server configs
    num_rounds: int = 10
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    local_epochs: int = 1
    checkpoint_dir: str = "checkpoints"
    strategy: str = "fedavg"
    client_config_files: Optional[List[str]] = None
    # Client configs
    client_id: Optional[int] = None
    local_epochs: Optional[int] = None
