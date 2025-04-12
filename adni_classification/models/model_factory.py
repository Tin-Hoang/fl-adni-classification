"""Model factory for ADNI classification."""

from typing import Dict, Any, Type

from adni_classification.models.base_model import BaseModel
from adni_classification.models.resnet3d import ResNet3D
from adni_classification.models.densenet3d import DenseNet3D
from adni_classification.models.simple_cnn import Simple3DCNN


class ModelFactory:
    """Factory class for creating model instances."""

    _models: Dict[str, Type[BaseModel]] = {
        "resnet3d": ResNet3D,
        "densenet3d": DenseNet3D,
        "simple3dcnn": Simple3DCNN,
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> BaseModel:
        """Create a model instance.

        Args:
            model_name: Name of the model to create
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            Model instance

        Raises:
            ValueError: If model_name is not supported
        """
        if model_name not in cls._models:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(cls._models.keys())}")

        return cls._models[model_name](**kwargs)

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model class.

        Args:
            name: Name of the model
            model_class: Model class to register
        """
        cls._models[name] = model_class
