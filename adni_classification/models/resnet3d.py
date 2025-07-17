"""3D ResNet model for ADNI classification."""

from typing import Dict, List, Optional

import monai.networks.nets as monai_nets
import torch

from adni_classification.models.base_model import BaseModel


class ResNet3D(BaseModel):
    """3D ResNet model for ADNI classification."""

    # ResNet configurations for different depths
    RESNET_CONFIGS: Dict[int, Dict[str, List[int]]] = {
        18: {"block": "basic", "layers": [2, 2, 2, 2], "block_inplanes": [64, 128, 256, 512]},
        34: {"block": "basic", "layers": [3, 4, 6, 3], "block_inplanes": [64, 128, 256, 512]},
        50: {"block": "bottleneck", "layers": [3, 4, 6, 3], "block_inplanes": [64, 128, 256, 512]},
        101: {"block": "bottleneck", "layers": [3, 4, 23, 3], "block_inplanes": [64, 128, 256, 512]},
        152: {"block": "bottleneck", "layers": [3, 8, 36, 3], "block_inplanes": [64, 128, 256, 512]},
    }

    def __init__(self, num_classes: int = 3, model_depth: int = 50, pretrained_checkpoint: Optional[str] = None):
        """Initialize ResNet3D model.

        Args:
            num_classes: Number of output classes
            model_depth: Depth of ResNet (18, 34, 50, 101, 152)
            pretrained_checkpoint: Path to pretrained weights file
        """
        super().__init__(num_classes)

        # Validate model depth
        if model_depth not in self.RESNET_CONFIGS:
            raise ValueError(f"Model depth {model_depth} not supported. Available depths: {list(self.RESNET_CONFIGS.keys())}")

        # Get configuration for the specified model depth
        config = self.RESNET_CONFIGS[model_depth]

        # Initialize MONAI ResNet with the appropriate configuration
        self.model = monai_nets.ResNet(
            block=config["block"],
            layers=config["layers"],
            block_inplanes=config["block_inplanes"],
            n_input_channels=1,
            shortcut_type="A",
            spatial_dims=3,
            num_classes=num_classes
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
