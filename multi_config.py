# Flower Multi-Machine Configuration Utilities
# This module provides utilities for working with the unified FL configuration

from typing import Dict, List, Any, Optional
from adni_classification.config.config import Config
from adni_classification.config.fl_config import MultiMachineConfig


def load_config_from_yaml(yaml_path: str) -> Config:
    """Load unified configuration from YAML file."""
    return Config.from_yaml(yaml_path)


def get_multi_machine_config(config: Config) -> Optional[MultiMachineConfig]:
    """Extract multi-machine configuration from unified config."""
    return config.fl.multi_machine


def get_server_config_dict(config: Config) -> Dict[str, Any]:
    """Get server configuration as dictionary for backward compatibility."""
    if config.fl.multi_machine:
        return config.fl.multi_machine.get_server_config_dict()
    return {}


def get_clients_config_dict(config: Config) -> List[Dict[str, Any]]:
    """Get clients configuration as list of dictionaries for backward compatibility."""
    if config.fl.multi_machine:
        return config.fl.multi_machine.get_clients_config_dict()
    return []
