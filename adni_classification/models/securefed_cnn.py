"""SecureFedCNN model for ADNI classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from adni_classification.models.base_model import BaseModel


class SecureFedCNN(BaseModel):
    """Secure Federated CNN model for ADNI classification."""

    def __init__(
        self,
        num_classes: int = 3,
        pretrained_checkpoint: str = None,
        input_size: list = [182, 218, 182],
        classification_mode: str = "CN_MCI_AD"
    ):
        """Initialize Secure Federated CNN model.

        Args:
            num_classes: Number of output classes (default: 3 for CN, MCI, AD)
            pretrained_checkpoint: Path to pretrained weights file
            input_size: Size of the input image (default: [182, 218, 182])
            classification_mode: Mode for classification, either "CN_MCI_AD" (3 classes) or "CN_AD" (2 classes)
        """
        # Determine the actual number of classes based on classification_mode
        if classification_mode == "CN_AD":
            actual_num_classes = 2
        else:  # Default: "CN_MCI_AD"
            actual_num_classes = 3

        # Override num_classes if classification_mode is specified
        if num_classes != actual_num_classes:
            print(f"Warning: Specified num_classes={num_classes} doesn't match {classification_mode} mode which requires {actual_num_classes} classes.")
            print(f"Setting num_classes to {actual_num_classes} for {classification_mode} mode.")
            num_classes = actual_num_classes

        super().__init__(num_classes)

        # Store input size and classification mode
        self.input_size = input_size if input_size else [182, 218, 182]
        self.classification_mode = classification_mode

        # Ensure input_size is a list of integers
        if isinstance(self.input_size, (list, tuple)) and len(self.input_size) == 3:
            self.input_size = [int(x) for x in self.input_size]

        # Print the input size for debugging
        print(f"SecureFedCNN initialized with input_size: {self.input_size}")
        print(f"Classification mode: {self.classification_mode} with {num_classes} output classes")

        # Define the first convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=0),  # 8 x 180 x 216 x 180
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 8 x 90 x 108 x 90
        )

        # Define the second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=0),  # 16 x 88 x 106 x 88
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=3)  # 16 x 29 x 35 x 29
        )

        # Define the third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=0),  # 32 x 27 x 33 x 27
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 32 x 13 x 16 x 13
        )

        # Define the fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=0),  # 64 x 11 x 14 x 11
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=3)  # 64 x 3 x 4 x 3
        )

        # Calculate the size of the flattened features
        # After conv blocks, for input [182, 218, 182]:
        # conv1: [91, 109, 91], conv2: [30, 36, 30], conv3: [15, 18, 15], conv4: [5, 6, 5]
        flat_size = [3, 4, 3]
        self.flat_features = 64 * flat_size[0] * flat_size[1] * flat_size[2]  # 64 * 3 * 4 * 3 = 2304

        print(f"Calculated flat features size: {self.flat_features} from input size {self.input_size}")

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

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
            if x.is_cuda:
                torch.cuda.synchronize()

        # Pass through convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Store original flattened size for debugging
        orig_shape = x.shape

        # Flatten the features
        x = x.view(x.size(0), -1)

        # Check the flattened size for debugging
        if x.size(1) != self.flat_features:
            actual_size = x.size(1)
            print(f"Warning: Flattened size {actual_size} doesn't match expected size {self.flat_features}")
            print(f"Original tensor shape after conv layers: {orig_shape}")

            # Dynamically adapt to the actual size if needed
            if not hasattr(self, 'adapted_fc1') or self.adapted_fc1.in_features != actual_size:
                print(f"Adapting fully connected layer to actual size: {actual_size}")
                self.adapted_fc1 = nn.Linear(actual_size, 128).to(x.device)
                if hasattr(self, 'fc1'):
                    if actual_size < self.flat_features:
                        self.adapted_fc1.weight.data[:, :actual_size] = self.fc1.weight.data[:, :actual_size]
                    else:
                        self.adapted_fc1.weight.data[:, :self.flat_features] = self.fc1.weight.data
                    self.adapted_fc1.bias.data = self.fc1.bias.data

            # Use the adapted layer
            x = self.dropout(x)
            x = F.relu(self.adapted_fc1(x))
        else:
            # Regular forward pass
            x = self.dropout(x)
            x = F.relu(self.fc1(x))

        # Apply dropout before second fully connected layer
        x = self.dropout(x)

        # Pass through second fully connected layer
        x = self.fc2(x)

        return x
