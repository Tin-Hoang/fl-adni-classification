"""Strategy factory for dynamic FL strategy loading."""

from typing import Any, Dict, Type

import torch
import torch.nn as nn

from adni_classification.config.config import Config

from .base import ClientStrategyBase, FLStrategyBase
from .fedavg import FedAvgClient, FedAvgStrategy
from .fedprox import FedProxClient, FedProxStrategy
from .secagg import SecAggClient, SecAggStrategy
from .secaggplus import SecAggPlusClient, SecAggPlusStrategy


class StrategyFactory:
    """Factory class for creating FL strategies."""

    # Registry of available strategies
    SERVER_STRATEGIES = {
        "fedavg": FedAvgStrategy,
        "fedprox": FedProxStrategy,
        "secagg": SecAggStrategy,
        "secagg+": SecAggPlusStrategy,
        "secaggplus": SecAggPlusStrategy,  # Alternative name
    }

    CLIENT_STRATEGIES = {
        "fedavg": FedAvgClient,
        "fedprox": FedProxClient,
        "secagg": SecAggClient,
        "secagg+": SecAggPlusClient,
        "secaggplus": SecAggPlusClient,  # Alternative name
    }

    @classmethod
    def create_server_strategy(
        self, strategy_name: str, config: Config, model: nn.Module, wandb_logger: Any = None, **kwargs
    ) -> FLStrategyBase:
        """Create a server-side FL strategy.

        Args:
            strategy_name: Name of the strategy to create
            config: Configuration object
            model: PyTorch model
            wandb_logger: Wandb logger instance
            **kwargs: Additional strategy-specific parameters

        Returns:
            Server strategy instance

        Raises:
            ValueError: If strategy name is not supported
        """
        if strategy_name not in self.SERVER_STRATEGIES:
            available = ", ".join(self.SERVER_STRATEGIES.keys())
            raise ValueError(f"Unsupported server strategy: {strategy_name}. Available: {available}")

        strategy_class = self.SERVER_STRATEGIES[strategy_name]

        # Get strategy-specific parameters from config
        strategy_params = self._get_strategy_params(strategy_name, config)
        strategy_params.update(kwargs)

        print(f"Creating {strategy_name} server strategy with params: {strategy_params}")

        return strategy_class(config=config, model=model, wandb_logger=wandb_logger, **strategy_params)

    @classmethod
    def create_client_strategy(
        self,
        strategy_name: str,
        config: Config,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        **kwargs,
    ) -> ClientStrategyBase:
        """Create a client-side FL strategy.

        Args:
            strategy_name: Name of the strategy to create
            config: Configuration object
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to use for computation
            scheduler: Learning rate scheduler (optional)
            **kwargs: Additional strategy-specific parameters

        Returns:
            Client strategy instance

        Raises:
            ValueError: If strategy name is not supported
        """
        if strategy_name not in self.CLIENT_STRATEGIES:
            available = ", ".join(self.CLIENT_STRATEGIES.keys())
            raise ValueError(f"Unsupported client strategy: {strategy_name}. Available: {available}")

        strategy_class = self.CLIENT_STRATEGIES[strategy_name]

        # Get strategy-specific parameters from config
        strategy_params = self._get_strategy_params(strategy_name, config)
        strategy_params.update(kwargs)

        print(f"Creating {strategy_name} client strategy with params: {strategy_params}")

        return strategy_class(
            config=config,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            **strategy_params,
        )

    @classmethod
    def _get_strategy_params(self, strategy_name: str, config: Config) -> Dict[str, Any]:
        """Get strategy-specific parameters from config.

        Args:
            strategy_name: Name of the strategy
            config: Configuration object

        Returns:
            Dictionary of strategy parameters
        """
        params = {}

        # Check if config has strategy-specific section
        if hasattr(config, "strategies") and hasattr(config.strategies, strategy_name):
            strategy_config = getattr(config.strategies, strategy_name)
            params = strategy_config.__dict__.copy()

        # Handle specific strategy parameters
        if strategy_name == "fedprox":
            params.setdefault("mu", getattr(config.fl, "fedprox_mu", 0.01))

        elif strategy_name == "secagg":
            params.setdefault("noise_multiplier", getattr(config.fl, "secagg_noise_multiplier", 0.1))
            params.setdefault("dropout_rate", getattr(config.fl, "secagg_dropout_rate", 0.0))

        elif strategy_name in ["secagg+", "secaggplus"]:
            # SecAgg+ parameters
            params.setdefault("num_shares", getattr(config.fl, "secagg_num_shares", 3))
            params.setdefault("reconstruction_threshold", getattr(config.fl, "secagg_reconstruction_threshold", 3))
            params.setdefault("max_weight", getattr(config.fl, "secagg_max_weight", 16777216))
            params.setdefault("timeout", getattr(config.fl, "secagg_timeout", None))
            params.setdefault("clipping_range", getattr(config.fl, "secagg_clipping_range", 1.0))
            params.setdefault("quantization_range", getattr(config.fl, "secagg_quantization_range", 2**20))

        return params

    @classmethod
    def get_available_strategies(self) -> Dict[str, Dict[str, Type]]:
        """Get all available strategies.

        Returns:
            Dictionary containing server and client strategies
        """
        return {"server": self.SERVER_STRATEGIES.copy(), "client": self.CLIENT_STRATEGIES.copy()}

    @classmethod
    def register_strategy(
        self,
        strategy_name: str,
        server_class: Type[FLStrategyBase] = None,
        client_class: Type[ClientStrategyBase] = None,
    ):
        """Register a new strategy.

        Args:
            strategy_name: Name of the strategy
            server_class: Server strategy class
            client_class: Client strategy class
        """
        if server_class:
            self.SERVER_STRATEGIES[strategy_name] = server_class
            print(f"Registered server strategy: {strategy_name}")

        if client_class:
            self.CLIENT_STRATEGIES[strategy_name] = client_class
            print(f"Registered client strategy: {strategy_name}")

    @classmethod
    def validate_strategy_config(self, strategy_name: str, config: Config) -> bool:
        """Validate strategy configuration.

        Args:
            strategy_name: Name of the strategy
            config: Configuration object

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        validator = StrategyConfigValidator()

        if strategy_name == "fedprox":
            return validator.validate_fedprox_config(config)
        elif strategy_name == "secagg":
            return validator.validate_secagg_config(config)
        elif strategy_name in ["secagg+", "secaggplus"]:
            return validator.validate_secaggplus_config(config)
        else:
            # No specific validation for other strategies
            return True


class StrategyConfigValidator:
    """Validator for strategy configurations."""

    @staticmethod
    def validate_fedprox_config(config: Config) -> bool:
        """Validate FedProx configuration.

        Args:
            config: Configuration object

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        mu = getattr(config.fl, "fedprox_mu", 0.01)
        if not isinstance(mu, (int, float)) or mu < 0:
            raise ValueError(f"FedProx mu must be a non-negative number, got: {mu}")
        return True

    @staticmethod
    def validate_secagg_config(config: Config) -> bool:
        """Validate SecAgg configuration.

        Args:
            config: Configuration object

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        noise_multiplier = getattr(config.fl, "secagg_noise_multiplier", 0.1)
        dropout_rate = getattr(config.fl, "secagg_dropout_rate", 0.0)

        if not isinstance(noise_multiplier, (int, float)) or noise_multiplier < 0:
            raise ValueError(f"SecAgg noise_multiplier must be a non-negative number, got: {noise_multiplier}")

        if not isinstance(dropout_rate, (int, float)) or not (0 <= dropout_rate <= 1):
            raise ValueError(f"SecAgg dropout_rate must be a number in [0, 1], got: {dropout_rate}")

        return True

    @staticmethod
    def validate_secaggplus_config(config: Config) -> bool:
        """Validate SecAgg+ configuration.

        Args:
            config: Configuration object

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        num_shares = getattr(config.fl, "secagg_num_shares", 3)
        reconstruction_threshold = getattr(config.fl, "secagg_reconstruction_threshold", 3)
        max_weight = getattr(config.fl, "secagg_max_weight", 16777216)
        timeout = getattr(config.fl, "secagg_timeout", None)
        clipping_range = getattr(config.fl, "secagg_clipping_range", 1.0)
        quantization_range = getattr(config.fl, "secagg_quantization_range", 2**20)

        if not isinstance(num_shares, (int, float)) or num_shares < 0:
            raise ValueError(f"SecAgg+ num_shares must be a non-negative number, got: {num_shares}")
        if not isinstance(reconstruction_threshold, (int, float)) or reconstruction_threshold < 0:
            raise ValueError(
                f"SecAgg+ reconstruction_threshold must be a non-negative number, got: {reconstruction_threshold}"
            )
        if not isinstance(max_weight, (int, float)) or max_weight < 0:
            raise ValueError(f"SecAgg+ max_weight must be a non-negative number, got: {max_weight}")
        if timeout is not None and not isinstance(timeout, (int, float)):
            raise ValueError(f"SecAgg+ timeout must be a number or None, got: {timeout}")
        if not isinstance(clipping_range, (int, float)) or clipping_range < 0:
            raise ValueError(f"SecAgg+ clipping_range must be a non-negative number, got: {clipping_range}")
        if not isinstance(quantization_range, (int, float)) or quantization_range < 0:
            raise ValueError(f"SecAgg+ quantization_range must be a non-negative number, got: {quantization_range}")

        return True
