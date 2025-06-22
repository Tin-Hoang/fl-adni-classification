"""Rosanna's 3D CNN model for ADNI classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import os

from adni_classification.models.base_model import BaseModel


class RosannaCNN(BaseModel):
    """Rosanna's 3D CNN model for ADNI classification.

    This model is based on the 3D CNN architecture from the ADNI pretrained model.
    It can be used for transfer learning and fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained_checkpoint: Optional[str] = None,
        freeze_encoder: bool = False,
        dropout: float = 0.0,
        input_channels: int = 1,
    ):
        """Initialize RosannaCNN model.

        Args:
            num_classes: Number of output classes
            pretrained_checkpoint: Path to pretrained checkpoint file
            freeze_encoder: Whether to freeze encoder layers for fine-tuning
            dropout: Dropout probability
            input_channels: Number of input channels (default: 1)
        """
        super().__init__(num_classes)

        self.dropout_p = dropout
        self.freeze_encoder = freeze_encoder

        # Define the CNN architecture (based on CNN_8CL_B configuration)
        self.out_channels = [8, 8, 16, 16, 32, 32, 64, 64]
        self.in_channels = [input_channels] + self.out_channels[:-1]
        self.n_conv = len(self.out_channels)
        self.kernels = [(3, 3, 3)] * self.n_conv
        self.pooling = [(4, 4, 4), (0, 0, 0), (3, 3, 3), (0, 0, 0), (2, 2, 2),
                        (0, 0, 0), (2, 2, 2), (0, 0, 0)]

        # Build convolutional layers
        self.embedding = nn.ModuleList()
        for i in range(self.n_conv):
            pad = tuple([int((k-1)/2) for k in self.kernels[i]])
            if self.pooling[i] != (0, 0, 0):
                self.embedding.append(nn.Sequential(
                    nn.Conv3d(in_channels=self.in_channels[i],
                             out_channels=self.out_channels[i],
                             kernel_size=self.kernels[i],
                             stride=(1, 1, 1),
                             padding=pad,
                             bias=False),
                    nn.BatchNorm3d(self.out_channels[i]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(self.pooling[i], stride=self.pooling[i])
                ))
            else:
                self.embedding.append(nn.Sequential(
                    nn.Conv3d(in_channels=self.in_channels[i],
                             out_channels=self.out_channels[i],
                             kernel_size=self.kernels[i],
                             stride=(1, 1, 1),
                             padding=pad,
                             bias=False),
                    nn.BatchNorm3d(self.out_channels[i]),
                    nn.ReLU(inplace=True)
                ))

        # Calculate feature size after convolutions
        # Based on the original architecture: input (73, 96, 96) -> output (256 features)
        self.feature_size = 256

        # Fully connected layers
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.classifier = nn.Linear(self.feature_size, num_classes)

        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self.load_pretrained_weights(pretrained_checkpoint)

        # Freeze encoder if requested
        if self.freeze_encoder:
            self.freeze_encoder_layers()

    def load_pretrained_weights(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading pretrained weights from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Print checkpoint metadata if available
        if 'pretrained_info' in checkpoint:
            print("Checkpoint metadata:")
            for key, value in checkpoint['pretrained_info'].items():
                print(f"  {key}: {value}")

        if 'val_acc' in checkpoint:
            print(f"Original validation accuracy: {checkpoint['val_acc']:.2f}%")

        # Load model state dict
        model_state_dict = checkpoint['model_state_dict']
        current_state_dict = self.state_dict()

        # Filter out weights that don't match (e.g., classifier layer for different num_classes)
        filtered_state_dict = {}
        for key, value in model_state_dict.items():
            if key in current_state_dict and current_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"Skipping layer {key} due to shape mismatch or non-existence")

        # Load the filtered weights
        self.load_state_dict(filtered_state_dict, strict=False)
        print(f"Successfully loaded {len(filtered_state_dict)} layers from checkpoint")

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, tuple]:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 1, depth, height, width)
            return_features: Whether to return intermediate features

        Returns:
            Output tensor of shape (batch_size, num_classes) or tuple of (output, features)
        """
        # Apply convolutional layers
        features = []
        out = self.embedding[0](x)
        if return_features:
            features.append(out)

        for i in range(1, len(self.embedding)):
            out = self.embedding[i](out)
            if return_features:
                features.append(out)

        # Flatten features
        out = out.view(out.size(0), -1)

        # Apply fully connected layers
        out = self.dropout(out)
        out = self.classifier(out)

        if return_features:
            return out, features
        else:
            return out

    def freeze_encoder_layers(self) -> None:
        """Freeze encoder layers for fine-tuning."""
        print("Freezing encoder layers...")
        for param in self.embedding.parameters():
            param.requires_grad = False
        print("Encoder layers frozen")

    def unfreeze_encoder_layers(self) -> None:
        """Unfreeze encoder layers."""
        print("Unfreezing encoder layers...")
        for param in self.embedding.parameters():
            param.requires_grad = True
        print("Encoder layers unfrozen")

    def get_feature_extractor(self) -> nn.Module:
        """Get the feature extractor part of the model (without classifier)."""
        return self.embedding


class RosannaCNNConfig:
    """Configuration class for RosannaCNN (equivalent to CNN_8CL_B)."""

    def __init__(self):
        self.input_dim = [73, 96, 96]
        self.out_channels = [8, 8, 16, 16, 32, 32, 64, 64]
        self.in_channels = [1] + [nch for nch in self.out_channels[:-1]]
        self.n_conv = len(self.out_channels)
        self.kernels = [(3, 3, 3)] * self.n_conv
        self.pooling = [(4, 4, 4), (0, 0, 0), (3, 3, 3), (0, 0, 0), (2, 2, 2),
                        (0, 0, 0), (2, 2, 2), (0, 0, 0)]

        # Compute final dimensions
        for i in range(self.n_conv):
            for d in range(3):
                if self.pooling[i][d] != 0:
                    self.input_dim[d] = self._compute_output_size(
                        self.input_dim[d], self.pooling[i][d], 0, self.pooling[i][d]
                    )

        out = self.input_dim[0] * self.input_dim[1] * self.input_dim[2]
        self.fweights = [self.out_channels[-1] * out, 2]
        self.dropout = 0.0

    def _compute_output_size(self, i: int, K: int, P: int, S: int) -> int:
        """Compute output size after convolution/pooling operation."""
        output_size = ((i - K + 2*P)/S) + 1
        return int(output_size)
