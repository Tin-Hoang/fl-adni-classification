import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SSHConfig:
    """SSH connection configuration for multi-machine FL."""

    timeout: int = 30
    banner_timeout: int = 30
    auth_timeout: int = 30


@dataclass
class ClientMachineConfig:
    """Individual client machine configuration."""

    host: str
    username: str
    password: Optional[str] = None
    partition_id: int = 0
    project_dir: Optional[str] = None
    config_file: Optional[str] = None
    sequential_experiment: bool = False
    train_sequential_labels: Optional[List[str]] = None
    val_sequential_labels: Optional[List[str]] = None


@dataclass
class ServerMachineConfig:
    """Server machine configuration."""

    host: str
    username: str
    password: Optional[str] = None
    port: int = 9092
    config_file: Optional[str] = None
    sequential_experiment: bool = False
    train_sequential_labels: Optional[List[str]] = None
    val_sequential_labels: Optional[List[str]] = None


@dataclass
class MultiMachineConfig:
    """Multi-machine configuration for distributed FL."""

    server: Optional[ServerMachineConfig] = None
    clients: List[ClientMachineConfig] = field(default_factory=list)
    project_dir: Optional[str] = None
    venv_path: Optional[str] = None
    venv_activate: Optional[str] = None
    ssh: SSHConfig = field(default_factory=SSHConfig)

    def get_server_config_dict(self) -> Dict[str, Any]:
        """Get server configuration as dictionary for backward compatibility."""
        if not self.server:
            return {}
        config_dict = {
            "host": self.server.host,
            "username": self.server.username,
            "password": self.server.password or os.getenv("FL_PASSWORD"),
            "port": self.server.port,
        }
        if self.server.config_file:
            config_dict["config_file"] = self.server.config_file
        if self.server.sequential_experiment:
            config_dict["sequential_experiment"] = self.server.sequential_experiment
        if self.server.train_sequential_labels:
            config_dict["train_sequential_labels"] = self.server.train_sequential_labels
        if self.server.val_sequential_labels:
            config_dict["val_sequential_labels"] = self.server.val_sequential_labels
        return config_dict

    def get_clients_config_dict(self) -> List[Dict[str, Any]]:
        """Get clients configuration as list of dictionaries for backward compatibility."""
        client_configs = []
        for client in self.clients:
            client_dict = {
                "host": client.host,
                "username": client.username,
                "password": client.password or os.getenv("FL_PASSWORD"),
                "project_dir": client.project_dir or self.project_dir,
                "partition_id": client.partition_id,
            }
            if client.config_file:
                client_dict["config_file"] = client.config_file
            if client.sequential_experiment:
                client_dict["sequential_experiment"] = client.sequential_experiment
            if client.train_sequential_labels:
                client_dict["train_sequential_labels"] = client.train_sequential_labels
            if client.val_sequential_labels:
                client_dict["val_sequential_labels"] = client.val_sequential_labels
            client_configs.append(client_dict)
        return client_configs


@dataclass
class FLConfig:
    """Federated Learning configuration."""

    num_rounds: int = 10
    strategy: str = "fedavg"  # fedavg, fedprox, secagg, secagg+, secaggplus
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

    # SecAgg (original - differential privacy) specific parameters
    secagg_noise_multiplier: float = 0.1
    secagg_dropout_rate: float = 0.0

    # SecAgg+ (real secure aggregation) specific parameters
    secagg_num_shares: int = 3  # Number of secret shares for each client
    secagg_reconstruction_threshold: int = 3  # Minimum shares needed for reconstruction
    secagg_max_weight: int = 16777216  # Maximum weight value (2^24)
    secagg_timeout: Optional[float] = 30.0  # Timeout for SecAgg operations (seconds)
    secagg_clipping_range: float = 1.0  # Range for gradient clipping
    secagg_quantization_range: int = 1048576  # Range for quantization (2^20)

    # Client ID (used for client applications)
    client_id: Optional[int] = None

    # Multi-machine configuration
    multi_machine: Optional[MultiMachineConfig] = None
