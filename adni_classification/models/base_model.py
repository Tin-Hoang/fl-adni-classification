"""Base model class for ADNI classification."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base model class for ADNI classification."""

    def __init__(self, num_classes: int = 3):
        """Initialize base model.

        Args:
            num_classes: Number of output classes (default: 3 for CN, MCI, AD)
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
