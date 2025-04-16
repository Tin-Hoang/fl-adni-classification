"""SecureFedCNN model for ADNI classification."""

import torch
import torch.nn as nn

from adni_classification.models.base_model import BaseModel


class SecureFedCNN(BaseModel):
    """CNN model for distinguishing between CN, AD, and MCI cases."""

    def __init__(self, num_classes: int = 3):
        """Initialize SecureFedCNN model.

        Args:
            num_classes: Number of output classes (default: 3 for CN, MCI, AD)
        """
        super().__init__(num_classes)

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(kernel_size=3, stride=3)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Linear(64 * 11 * 11 * 11, 128)  # Adjusted for input size after conv blocks
        self.fc2 = nn.Linear(128, num_classes)  # Output classes: CN, MCI, AD

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 1, depth, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.conv_block1(x)
        x = self.relu(x)

        x = self.conv_block2(x)
        x = self.relu(x)

        x = self.conv_block3(x)
        x = self.relu(x)

        x = self.conv_block4(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
