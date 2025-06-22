"""Utilities for pretrained CNN model preprocessing and normalization."""

import torch
import numpy as np
from torchvision import transforms
from scipy import ndimage
from typing import Union, Tuple, List


def normalize_intensity(img_tensor: torch.Tensor, normalization: str = "mean") -> torch.Tensor:
    """Normalize image intensity.

    Args:
        img_tensor: Input image tensor
        normalization: Type of normalization ("mean" or "max")

    Returns:
        Normalized image tensor
    """
    if normalization == "mean":
        # Use non-zero voxels only for mean normalization
        mask = img_tensor.ne(0.0)
        if mask.sum() > 0:  # Check if there are non-zero values
            desired = img_tensor[mask]
            mean_val, std_val = desired.mean(), desired.std()
            if std_val > 0:  # Avoid division by zero
                img_tensor = (img_tensor - mean_val) / std_val
            else:
                img_tensor = img_tensor - mean_val

    elif normalization == "max":
        MAX, MIN = img_tensor.max(), img_tensor.min()
        if MAX > MIN:  # Avoid division by zero
            img_tensor = (img_tensor - MIN) / (MAX - MIN)
        else:
            img_tensor = img_tensor - MIN

    return img_tensor


def resize_data_volume_by_scale(data: np.ndarray, scale: Union[float, List[float]]) -> np.ndarray:
    """Resize the data based on the provided scale.

    Args:
        data: Input 3D numpy array
        scale: Scale factor(s) - float for uniform scaling or list of 3 floats

    Returns:
        Resized data volume
    """
    if isinstance(scale, float):
        scale_list = [scale, scale, scale]
    else:
        scale_list = scale
    return ndimage.interpolation.zoom(data, scale_list, order=0)


def img_processing(
    image: np.ndarray,
    scaling: float = 0.5,
    final_size: List[int] = [96, 96, 73]
) -> np.ndarray:
    """Process image with scaling and resizing.

    Args:
        image: Input 3D image array
        scaling: Initial scaling factor
        final_size: Target final size [width, height, depth]

    Returns:
        Processed image
    """
    # First resize with scaling factor
    image = resize_data_volume_by_scale(image, scale=scaling)

    # Then resize to final size
    new_scaling = [final_size[i] / image.shape[i] for i in range(3)]
    final_image = resize_data_volume_by_scale(image, scale=new_scaling)

    return final_image


def torch_norm(input_image: np.ndarray) -> torch.Tensor:
    """Convert numpy image to normalized tensor.

    Args:
        input_image: Input image as numpy array

    Returns:
        Normalized tensor with batch dimension
    """
    # Convert to tensor
    if isinstance(input_image, np.ndarray):
        input_tensor = torch.from_numpy(input_image).float()
    else:
        input_tensor = input_image

    # Add channel dimension if needed
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add channel dimension

    # Normalize
    input_tensor = normalize_intensity(input_tensor)

    # Add batch dimension if needed
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    return input_tensor


class PretrainedDataTransform:
    """Transform class for pretrained model data preprocessing."""

    def __init__(
        self,
        scaling: float = 0.5,
        final_size: List[int] = [96, 96, 73],
        normalization: str = "mean",
        preprocessing: bool = True
    ):
        """Initialize transform.

        Args:
            scaling: Initial scaling factor
            final_size: Target final size
            normalization: Normalization method
            preprocessing: Whether to apply image processing
        """
        self.scaling = scaling
        self.final_size = final_size
        self.normalization = normalization
        self.preprocessing = preprocessing

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply transform to image.

        Args:
            image: Input image

        Returns:
            Transformed tensor
        """
        if self.preprocessing:
            image = img_processing(image, self.scaling, self.final_size)

        return torch_norm(image)
