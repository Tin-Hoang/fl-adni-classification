"""Simple 3D CNN model for ADNI classification."""

import torch
import torch.nn as nn


class Simple3DCNN(nn.Module):
    """A simple 3D CNN model for brain MRI classification.
    
    This model is designed to be lightweight and suitable for small datasets.
    It uses a simple architecture with 3 convolutional blocks followed by
    global average pooling and a fully connected layer.
    """
    
    def __init__(self, num_classes=3, dropout_rate=0.5):
        """Initialize the model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Dropout3d(dropout_rate),
            
            # Second conv block
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Dropout3d(dropout_rate),
            
            # Third conv block
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Dropout3d(dropout_rate),
            
            # Global average pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, depth, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x 