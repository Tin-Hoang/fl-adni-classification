"""Rosanna's 3D CNN model for ADNI classification."""

import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        input_size: tuple = (73, 96, 96),
    ):
        """Initialize RosannaCNN model.

        Args:
            num_classes: Number of output classes
            pretrained_checkpoint: Path to pretrained checkpoint file
            freeze_encoder: Whether to freeze encoder layers for fine-tuning
            dropout: Dropout probability
            input_channels: Number of input channels (default: 1)
            input_size: Input image dimensions (D, H, W) - default: (73, 96, 96)
        """
        super().__init__(num_classes)
        print(f"Initializing RosannaCNN with num_classes={num_classes}, pretrained_checkpoint={pretrained_checkpoint}, freeze_encoder={freeze_encoder}, dropout={dropout}, input_channels={input_channels}")
        self.dropout_p = dropout
        self.freeze_encoder = freeze_encoder
        self.input_size = list(input_size)

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

        # Calculate feature size after convolutions (matching original implementation)
        self.feature_size = self._calculate_feature_size()

        # Fully connected layers (matching original architecture)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_p)

        # Build fully connected layers - original uses fweights = [feature_size, num_classes]
        self.f = nn.ModuleList()
        fweights = [self.feature_size, num_classes]
        for i in range(len(fweights)-1):
            self.f.append(nn.Linear(fweights[i], fweights[i+1]))
        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self.load_pretrained_weights(pretrained_checkpoint)

        # Freeze encoder if requested
        if self.freeze_encoder:
            self.freeze_encoder_layers()

    def _calculate_feature_size(self) -> int:
        """Calculate the feature size after all convolutional and pooling operations.

        This matches the original implementation's dynamic calculation.
        """
        # Start with input dimensions
        current_dims = self.input_size.copy()

        # Apply pooling operations to calculate final dimensions
        for i in range(self.n_conv):
            for d in range(3):
                if self.pooling[i][d] != 0:
                    current_dims[d] = self._compute_output_size(
                        current_dims[d], self.pooling[i][d], 0, self.pooling[i][d]
                    )

        # Final feature size = channels * depth * height * width
        feature_size = self.out_channels[-1] * current_dims[0] * current_dims[1] * current_dims[2]
        print(f"Calculated feature size: {feature_size} (dims: {current_dims})")
        return feature_size

    def _compute_output_size(self, i: int, K: int, P: int, S: int) -> int:
        """Compute output size after convolution/pooling operation."""
        output_size = ((i - K + 2*P)/S) + 1
        return int(output_size)

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

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # If it's a full checkpoint with metadata
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']

                # Print checkpoint metadata if available
                if 'pretrained_info' in checkpoint:
                    print("Checkpoint metadata:")
                    for key, value in checkpoint['pretrained_info'].items():
                        print(f"  {key}: {value}")

                if 'val_acc' in checkpoint:
                    print(f"Original validation accuracy: {checkpoint['val_acc']:.2f}%")
            else:
                # If it's just the state dict
                model_state_dict = checkpoint
        else:
            # If it's a raw state dict
            model_state_dict = checkpoint

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

    def forward(self, x: torch.Tensor, return_features: bool = False, return_softmax: bool = False) -> Union[torch.Tensor, tuple]:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 1, depth, height, width)
            return_features: Whether to return intermediate features
            return_softmax: Whether to return softmax probabilities (like original)

        Returns:
            Output tensor of shape (batch_size, num_classes) or tuple of (output, features)
        """
        # Apply convolutional layers (matching original implementation)
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

        # Apply fully connected layers (matching original architecture)
        for fc in self.f[:-1]:
            out = fc(out)
            out = self.relu(out)
            out = self.dropout(out)

        # Final layer (no activation applied here in original)
        out = self.f[-1](out)

        # Apply softmax if requested (matching original behavior)
        if return_softmax:
            out = F.softmax(out, dim=1)

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

    def extract_features_at_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Extract features at a specific convolutional layer.

        Args:
            x: Input tensor
            layer_idx: Index of the layer to extract features from (0-based)

        Returns:
            Features from the specified layer
        """
        if layer_idx >= len(self.embedding):
            raise ValueError(f"Layer index {layer_idx} out of range. Model has {len(self.embedding)} layers.")

        out = x
        for i in range(layer_idx + 1):
            out = self.embedding[i](out)

        return out.view(out.size(0), -1)


class RosannaCNNConfig:
    """Configuration class for RosannaCNN (equivalent to CNN_8CL_B)."""

    def __init__(self, input_size: tuple = (73, 96, 96), num_classes: int = 2):
        self.input_dim = list(input_size)
        self.out_channels = [8, 8, 16, 16, 32, 32, 64, 64]
        self.in_channels = [1] + self.out_channels[:-1]
        self.n_conv = len(self.out_channels)
        self.kernels = [(3, 3, 3)] * self.n_conv
        self.pooling = [(4, 4, 4), (0, 0, 0), (3, 3, 3), (0, 0, 0), (2, 2, 2),
                        (0, 0, 0), (2, 2, 2), (0, 0, 0)]

        # Compute final dimensions (matching original implementation)
        for i in range(self.n_conv):
            for d in range(3):
                if self.pooling[i][d] != 0:
                    self.input_dim[d] = self._compute_output_size(
                        self.input_dim[d], self.pooling[i][d], 0, self.pooling[i][d]
                    )

        out = self.input_dim[0] * self.input_dim[1] * self.input_dim[2]
        self.fweights = [self.out_channels[-1] * out, num_classes]
        self.dropout = 0.0

    def _compute_output_size(self, i: int, K: int, P: int, S: int) -> int:
        """Compute output size after convolution/pooling operation."""
        output_size = ((i - K + 2*P)/S) + 1
        return int(output_size)
