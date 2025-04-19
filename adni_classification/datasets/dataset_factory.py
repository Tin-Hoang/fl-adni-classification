"""Factory module for creating ADNI datasets."""

from typing import Dict, Any, Union, Optional, Tuple
import torch
import monai

from adni_classification.datasets.adni_smartcache_dataset import ADNISmartCacheDataset
from adni_classification.datasets.adni_cache_dataset import ADNICacheDataset
from adni_classification.datasets.adni_dataset import ADNIDataset
from adni_classification.datasets.transforms import get_transforms


def create_adni_dataset(
    dataset_type: str = "smartcache",
    csv_path: str = "",
    img_dir: str = "",
    transform: Optional[monai.transforms.Compose] = None,
    cache_rate: float = 1.0,
    num_workers: int = 0,
    replace_rate: float = 0.1,
    cache_num: Optional[int] = None,
    **kwargs: Any
) -> Union[ADNISmartCacheDataset, ADNICacheDataset, ADNIDataset]:
    """Create a dataset instance based on the specified type.

    Args:
        dataset_type: Type of dataset to create ('smartcache', 'cache', or 'normal')
        csv_path: Path to the CSV file containing image metadata and labels
        img_dir: Path to the directory containing the image files
        transform: Optional transform to apply to the images
        cache_rate: The percentage of data to be cached (default: 1.0 = 100%)
        num_workers: Number of subprocesses to use for data loading (default: 0)
        replace_rate: Rate to randomly replace items in cache with new items (default: 0.1)
                      Only used for SmartCacheDataset
        cache_num: Number of items to cache. Default: None (cache_rate * len(data))
        **kwargs: Additional arguments to pass to the dataset constructor

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset_type is not supported
    """
    print(f"Creating ADNI dataset of type: {dataset_type}")

    if dataset_type.lower() == "smartcache":
        return ADNISmartCacheDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
            replace_rate=replace_rate,
            cache_num=cache_num,
            **kwargs
        )
    elif dataset_type.lower() == "cache":
        # Note: CacheDataset doesn't use replace_rate, but we ignore it here for API compatibility
        return ADNICacheDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
            cache_num=cache_num,
            **kwargs
        )
    elif dataset_type.lower() == "normal":
        return ADNIDataset(
            csv_path=csv_path,
            img_dir=img_dir,
            transform=transform,
            **kwargs  # Pass any additional arguments
        )
    else:
        raise ValueError(f"Dataset type '{dataset_type}' not supported. Available types: 'smartcache', 'cache', 'normal'")


def get_transforms_from_config(
    config: Union[Dict[str, Any], Any],
    mode: str = "train",
    device: Optional[torch.device] = None
) -> monai.transforms.Compose:
    """Get transforms based on configuration.

    Args:
        config: Configuration dictionary or DataConfig object for transforms
        mode: Either "train" or "val"
        device: Device to use for transforms (default: None, will use CPU)

    Returns:
        A Compose transform
    """
    # Handle both dictionary and DataConfig objects
    if isinstance(config, dict):
        resize_size = config.get("resize_size", (160, 160, 160))
        resize_mode = config.get("resize_mode", "trilinear")
        use_spacing = config.get("use_spacing", True)
        spacing_size = config.get("spacing_size", (1, 1, 1))
        transform_device = config.get("transform_device")
    else:
        # It's a DataConfig-like object - use duck typing
        resize_size = getattr(config, "resize_size", (160, 160, 160))
        resize_mode = getattr(config, "resize_mode", "trilinear")
        use_spacing = getattr(config, "use_spacing", True)
        spacing_size = getattr(config, "spacing_size", (1, 1, 1))
        transform_device = getattr(config, "transform_device", None)

    # Get device from config if not explicitly provided
    if device is None and transform_device is not None:
        device = torch.device(transform_device)

    return get_transforms(
        mode=mode,
        resize_size=resize_size,
        resize_mode=resize_mode,
        use_spacing=use_spacing,
        spacing_size=spacing_size,
        device=device
    )
