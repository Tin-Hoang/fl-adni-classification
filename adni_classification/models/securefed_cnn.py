"""SecureFedCNN model for ADNI classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from adni_classification.models.base_model import BaseModel


class SecureFedCNN(BaseModel):
    """Secure Federated CNN model for ADNI classification."""

    def __init__(
        self,
        num_classes: int = 3,
        pretrained_checkpoint: str = None,
        input_size: list = [128, 128, 128]
    ):
        """Initialize Secure Federated CNN model.

        Args:
            num_classes: Number of output classes
            pretrained_checkpoint: Path to pretrained weights file
            input_size: Size of the input image (default: [128, 128, 128])
        """
        super().__init__(num_classes)

        # Store input size
        self.input_size = input_size if input_size else [128, 128, 128]

        # Ensure input_size is a list of integers
        if isinstance(self.input_size, (list, tuple)) and len(self.input_size) == 3:
            self.input_size = [int(x) for x in self.input_size]

        # Print the input size for debugging
        print(f"SecureFedCNN initialized with input_size: {self.input_size}")

        # Define the first convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Define the second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Define the third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Define the fourth convolutional block (secure)
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Calculate the size of the flattened features
        # After 4 MaxPool layers with stride 2, the spatial dimensions are reduced by a factor of 2^4 = 16
        flat_size = [dim // 16 for dim in self.input_size]
        self.flat_features = 256 * flat_size[0] * flat_size[1] * flat_size[2]

        print(f"Calculated flat features size: {self.flat_features} from input size {self.input_size}")

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self.load_pretrained_weights(pretrained_checkpoint)

    def load_pretrained_weights(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 1, depth, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Check input shape for debugging
        if x.shape[2:] != torch.Size(self.input_size):
            print(f"Warning: Input shape {x.shape[2:]} doesn't match expected shape {self.input_size}")

        # Pass through convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Flatten the features
        x = x.view(x.size(0), -1)

        # Check the flattened size for debugging
        if x.size(1) != self.flat_features:
            print(f"Warning: Flattened size {x.size(1)} doesn't match expected size {self.flat_features}")
            # Dynamically adapt to the actual size if needed
            if not hasattr(self, 'adapted_fc1') or self.adapted_fc1.in_features != x.size(1):
                print(f"Adapting fully connected layer to actual size: {x.size(1)}")
                self.adapted_fc1 = nn.Linear(x.size(1), 512).to(x.device)
                # Initialize weights with existing weights if possible
                if x.size(1) < self.flat_features:
                    self.adapted_fc1.weight.data[:, :x.size(1)] = self.fc1.weight.data[:, :x.size(1)]
                else:
                    self.adapted_fc1.weight.data = self.fc1.weight.data
                self.adapted_fc1.bias.data = self.fc1.bias.data
            x = F.relu(self.adapted_fc1(x))
        else:
            # Regular forward pass
            x = F.relu(self.fc1(x))

        # Pass through second fully connected layer
        x = self.fc2(x)

        return x
