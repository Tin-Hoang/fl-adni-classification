"""3D DenseNet model for ADNI classification."""

from typing import Optional, Tuple
import torch
import monai.networks.nets as monai_nets

from adni_classification.models.base_model import BaseModel


class DenseNet3D(BaseModel):
    """3D DenseNet model for ADNI classification."""

    def __init__(self, num_classes: int = 3, growth_rate: int = 32, block_config: Tuple[int, ...] = (6, 12, 24, 16), pretrained: bool = False, weights_path: Optional[str] = None):
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
