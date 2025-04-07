"""Model factory for ADNI classification."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional

import torch
import torch.nn as nn
import monai.networks.nets as monai_nets
from adni_classification.models.simple_cnn import Simple3DCNN


class BaseModel(nn.Module, ABC):
    """Base model class for ADNI classification."""

    def __init__(self, num_classes: int = 3):
        """Initialize base model.

        Args:
            num_classes: Number of output classes (default: 3 for AD, MCI, NC)
        """
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass


class ResNet3D(BaseModel):
    """3D ResNet model for ADNI classification."""

    def __init__(self, num_classes: int = 3, model_depth: int = 50, pretrained: bool = False, weights_path: Optional[str] = None):
        """Initialize ResNet3D model.

        Args:
            num_classes: Number of output classes
            model_depth: Depth of ResNet (18, 34, 50, 101, 152)
            pretrained: Whether to use pretrained weights
            weights_path: Path to pretrained weights file
        """
        super().__init__(num_classes)

        # Initialize MONAI ResNet
        self.model = monai_nets.ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            n_input_channels=1,
            shortcut_type="A",
            spatial_dims=3,
            n_classes=num_classes
        )

        # Load pretrained weights if specified
        if pretrained and weights_path:
            self.load_pretrained_weights(weights_path)

    def load_pretrained_weights(self, weights_path: str) -> None:
        """Load pretrained weights from file.

        Args:
            weights_path: Path to pretrained weights file
        """
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {weights_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 1, depth, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)


class DenseNet3D(BaseModel):
    """3D DenseNet model for ADNI classification."""

    def __init__(self, num_classes: int = 3, growth_rate: int = 32, block_config: tuple = (6, 12, 24, 16), pretrained: bool = False, weights_path: Optional[str] = None):
        """Initialize DenseNet3D model.

        Args:
            num_classes: Number of output classes
            growth_rate: Growth rate for DenseNet
            block_config: Number of layers in each dense block
            pretrained: Whether to use pretrained weights
            weights_path: Path to pretrained weights file
        """
        super().__init__(num_classes)

        # Initialize MONAI DenseNet
        self.model = monai_nets.DenseNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            growth_rate=growth_rate,
            block_config=block_config,
            dropout_prob=0.2
        )

        # Load pretrained weights if specified
        if pretrained and weights_path:
            self.load_pretrained_weights(weights_path)

    def load_pretrained_weights(self, weights_path: str) -> None:
        """Load pretrained weights from file.

        Args:
            weights_path: Path to pretrained weights file
        """
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {weights_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 1, depth, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)


class ModelFactory:
    """Factory class for creating model instances."""

    _models: Dict[str, Type[BaseModel]] = {
        "resnet3d": ResNet3D,
        "densenet3d": DenseNet3D,
        "simple3dcnn": Simple3DCNN,
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> BaseModel:
        """Create a model instance.

        Args:
            model_name: Name of the model to create
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            Model instance

        Raises:
            ValueError: If model_name is not supported
        """
        if model_name not in cls._models:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(cls._models.keys())}")

        return cls._models[model_name](**kwargs)

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model class.

        Args:
            name: Name of the model
            model_class: Model class to register
        """
        cls._models[name] = model_class
