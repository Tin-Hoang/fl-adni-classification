"""Configuration management for ADNI classification."""

import os
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any

import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    train_csv_path: str
    val_csv_path: str
    img_dir: str
    dataset_type: str = "normal"  # Options: "smartcache", "cache", "normal", "persistent"
    resize_size: List[int] = field(default_factory=lambda: [160, 160, 160])
    resize_mode: str = "trilinear"
    use_spacing: bool = False
    spacing_size: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    cache_rate: float = 1.0  # Percentage of data to cache (0.0-1.0)
    cache_num_workers: int = 8  # Number of workers for CacheDataset initialization
    cache_dir: str = "./persistent_cache"  # Directory to store the persistent cache (for PersistentDataset)
    use_multiprocessing_transforms: bool = False  # Whether to use multiprocessing-safe transforms
    transform_device: Optional[str] = None  # Device to use for transforms (e.g., "cuda" or "cpu")
    multiprocessing_context: str = "spawn"  # Options: "spawn", "fork", "forkserver"
    classification_mode: str = "CN_MCI_AD"  # Mode for classification, either "CN_MCI_AD" (3 classes) or "CN_AD" (2 classes)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    num_classes: int = 3
    pretrained_checkpoint: Optional[str] = None
    # ResNet specific parameters
    model_depth: Optional[int] = None
    # DenseNet specific parameters
    growth_rate: Optional[int] = None
    block_config: Optional[Tuple[int, ...]] = None


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    save_best: bool = True
    save_latest: bool = True
    save_regular: bool = True
    save_frequency: int = 1  # Save a regular checkpoint every N epochs


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    output_dir: str
    seed: int = 42  # Random seed for reproducibility
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    visualize: bool = False
    lr_scheduler: str = "plateau"
    val_epoch_freq: int = 5  # Run validation every N epochs
    use_class_weights: bool = False
    class_weight_type: str = "inverse"  # Options: "inverse", "sqrt_inverse", "effective", "manual"
    manual_class_weights: Optional[List[float]] = None  # Manual class weights if class_weight_type is "manual"
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    use_wandb: bool
    project: str
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    run_name: str = ""


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config object from a dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))

        # Handle the checkpoint config if present
        training_dict = config_dict.get("training", {})
        checkpoint_dict = training_dict.pop("checkpoint", {}) if "checkpoint" in training_dict else {}
        checkpoint_config = CheckpointConfig(**checkpoint_dict)

        # Create training config with the checkpoint config
        training_config = TrainingConfig(**training_dict, checkpoint=checkpoint_config)

        wandb_config = WandbConfig(**config_dict.get("wandb", {}))

        config = cls(
            data=data_config,
            model=model_config,
            training=training_config,
            wandb=wandb_config,
        )

        # Post-process: Generate output directory based on run_name and timestamp
        config._post_process()

        return config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Create a Config object from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def _post_process(self) -> None:
        """Post-process the configuration.

        This includes:
        - Generating a unique output directory based on run_name and timestamp
        - If no run_name is specified, use the model name and timestamp
        - If run_name is specified, append a timestamp to it
        """
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate run_name if not provided or append timestamp if provided
        if not self.wandb.run_name:
            model_identifier = f"{self.model.name}"
            if self.model.name == "resnet3d" and self.model.model_depth:
                model_identifier = f"{self.model.name}{self.model.model_depth}"
            self.wandb.run_name = f"{model_identifier}_{timestamp}"
        else:
            # Append timestamp to existing run_name
            self.wandb.run_name = f"{self.wandb.run_name}_{timestamp}"

        # Update output directory to include run_name and timestamp if it doesn't already
        if self.training.output_dir == "outputs" or not self.training.output_dir:
            self.training.output_dir = os.path.join("outputs", f"{self.wandb.run_name}")
        elif os.path.basename(self.training.output_dir) != self.wandb.run_name:
            # If output directory is specified but doesn't match run_name, append run_name
            self.training.output_dir = os.path.join(self.training.output_dir, f"{self.wandb.run_name}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Config object to a dictionary."""
        return {
            "data": {
                "train_csv_path": self.data.train_csv_path,
                "val_csv_path": self.data.val_csv_path,
                "img_dir": self.data.img_dir,
                "dataset_type": self.data.dataset_type,
                "resize_size": self.data.resize_size,
                "resize_mode": self.data.resize_mode,
                "use_spacing": self.data.use_spacing,
                "spacing_size": self.data.spacing_size,
                "cache_rate": self.data.cache_rate,
                "cache_num_workers": self.data.cache_num_workers,
                "cache_dir": self.data.cache_dir,
                "use_multiprocessing_transforms": self.data.use_multiprocessing_transforms,
                "transform_device": self.data.transform_device,
                "multiprocessing_context": self.data.multiprocessing_context,
                "classification_mode": self.data.classification_mode,
            },
            "model": {
                "name": self.model.name,
                "num_classes": self.model.num_classes,
                "model_depth": self.model.model_depth,
                "growth_rate": self.model.growth_rate,
                "block_config": self.model.block_config,
                "pretrained_checkpoint": self.model.pretrained_checkpoint,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "num_epochs": self.training.num_epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "num_workers": self.training.num_workers,
                "output_dir": self.training.output_dir,
                "seed": self.training.seed,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "mixed_precision": self.training.mixed_precision,
                "visualize": self.training.visualize,
                "lr_scheduler": self.training.lr_scheduler,
                "val_epoch_freq": self.training.val_epoch_freq,
                "use_class_weights": self.training.use_class_weights,
                "class_weight_type": self.training.class_weight_type,
                "manual_class_weights": self.training.manual_class_weights,
                "checkpoint": {
                    "save_best": self.training.checkpoint.save_best,
                    "save_latest": self.training.checkpoint.save_latest,
                    "save_regular": self.training.checkpoint.save_regular,
                    "save_frequency": self.training.checkpoint.save_frequency,
                },
            },
            "wandb": {
                "use_wandb": self.wandb.use_wandb,
                "project": self.wandb.project,
                "entity": self.wandb.entity,
                "tags": self.wandb.tags,
                "notes": self.wandb.notes,
                "run_name": self.wandb.run_name,
            },
        }

    def to_yaml(self, yaml_path: str) -> None:
        """Save the Config object to a YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
