"""Configuration management for ADNI classification."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any

import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    train_csv_path: str
    val_csv_path: str
    img_dir: str
    resize_size: List[int] = field(default_factory=lambda: [224, 224, 224])
    resize_mode: str = "trilinear"


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    num_classes: int = 3
    pretrained: bool = False
    weights_path: Optional[str] = None
    # ResNet specific parameters
    model_depth: Optional[int] = None
    # DenseNet specific parameters
    growth_rate: Optional[int] = None
    block_config: Optional[Tuple[int, ...]] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    output_dir: str
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False


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
        training_config = TrainingConfig(**config_dict.get("training", {}))
        wandb_config = WandbConfig(**config_dict.get("wandb", {}))

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            wandb=wandb_config,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Create a Config object from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Config object to a dictionary."""
        return {
            "data": {
                "train_csv_path": self.data.train_csv_path,
                "val_csv_path": self.data.val_csv_path,
                "img_dir": self.data.img_dir,
                "resize_size": self.data.resize_size,
                "resize_mode": self.data.resize_mode,
            },
            "model": {
                "name": self.model.name,
                "num_classes": self.model.num_classes,
                "model_depth": self.model.model_depth,
                "growth_rate": self.model.growth_rate,
                "block_config": self.model.block_config,
                "pretrained": self.model.pretrained,
                "weights_path": self.model.weights_path,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "num_epochs": self.training.num_epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "num_workers": self.training.num_workers,
                "output_dir": self.training.output_dir,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "mixed_precision": self.training.mixed_precision,
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
