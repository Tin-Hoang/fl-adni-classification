"""3D DenseNet model for ADNI classification."""

from typing import Optional, Tuple

import monai.networks.nets as monai_nets
import torch

from adni_classification.models.base_model import BaseModel


class DenseNet3D(BaseModel):
    """3D DenseNet model for ADNI classification."""

    def __init__(self, num_classes: int = 3, growth_rate: int = 32, block_config: Tuple[int, ...] = (6, 12, 24, 16), pretrained_checkpoint: Optional[str] = None):
        """Initialize DenseNet3D model.

        Args:
            num_classes: Number of output classes
            growth_rate: Growth rate for DenseNet
            block_config: Number of layers in each dense block
            pretrained_checkpoint: Path to pretrained weights file
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
        if pretrained_checkpoint:
            self.load_pretrained_weights(pretrained_checkpoint)

    def load_pretrained_weights(self, pretrained_checkpoint: str) -> None:
        """Load pretrained weights from file.

        Args:
            pretrained_checkpoint: Path to pretrained weights file
        """
        state_dict = torch.load(pretrained_checkpoint, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_checkpoint}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 1, depth, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
