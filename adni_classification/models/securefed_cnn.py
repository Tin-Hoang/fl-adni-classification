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
        # Print the model configuration for clarity
        print(f"SecureFedCNN initializing with:")
        print(f"- num_classes: {num_classes}")
        print(f"- classification_mode: {classification_mode}")

        # For backwards compatibility only - don't force override num_classes
        # Only print warning if there's an apparent mismatch
        expected_classes = 2 if classification_mode == "CN_AD" else 3
        if num_classes != expected_classes:
            print(f"Note: Using {num_classes} output classes with {classification_mode} mode.")
            print(f"This is different from the default of {expected_classes} classes for {classification_mode} mode.")

        # Initialize the base model with the specified number of classes
        super().__init__(num_classes)

        # Store classification mode for reference
        self.classification_mode = classification_mode
        print(f"Classification mode: {classification_mode} with {num_classes} output classes")

        # Store input size and classification mode
        self.input_size = input_size if input_size else [182, 218, 182]

        # Ensure input_size is a list of integers
        if isinstance(self.input_size, (list, tuple)) and len(self.input_size) == 3:
            self.input_size = [int(x) for x in self.input_size]

        # Print the input size for debugging
        print(f"SecureFedCNN initialized with input_size: {self.input_size}")

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

        # Dynamically calculate the size of the flattened features
        self.flat_features = self._calculate_flat_features()
        print(f"Calculated flat features size: {self.flat_features} from input size {self.input_size}")

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self.load_pretrained_weights(pretrained_checkpoint)

    def _calculate_flat_features(self) -> int:
        """Calculate the number of flat features after convolutional layers.

        Returns:
            Number of features after flattening the conv output
        """
        # Create a dummy input with the specified input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.input_size)

            # Pass through convolutional layers
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)

            # Calculate the flattened size
            flat_size = x.view(1, -1).size(1)

        return flat_size

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

        # Flatten the features
        x = x.view(x.size(0), -1)

        # Check the flattened size for debugging
        if x.size(1) != self.flat_features:
            actual_size = x.size(1)
            print(f"Warning: Flattened size {actual_size} doesn't match expected size {self.flat_features}")
            print(f"This may indicate a mismatch between initialization and runtime input sizes.")

            # For safety, we'll handle this gracefully but it shouldn't happen with proper initialization
            raise RuntimeError(
                f"Input size mismatch: expected flattened size {self.flat_features}, "
                f"got {actual_size}. This suggests the input shape {x.shape[2:]} differs from "
                f"the initialization input shape {self.input_size}."
            )

        # Pass through fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
