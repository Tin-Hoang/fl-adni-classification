"""Model factory for ADNI classification."""

from typing import Dict, Any, Type

from adni_classification.models.base_model import BaseModel
from adni_classification.models.resnet3d import ResNet3D
from adni_classification.models.densenet3d import DenseNet3D
from adni_classification.models.simple_cnn import Simple3DCNN
from adni_classification.models.securefed_cnn import SecureFedCNN


class ModelFactory:
    """Factory class for creating model instances."""

    _models: Dict[str, Type[BaseModel]] = {
        "resnet3d": ResNet3D,
        "densenet3d": DenseNet3D,
        "simple3dcnn": Simple3DCNN,
        "securefed_cnn": SecureFedCNN,
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

        # Extract input_size from kwargs if specified in config for SecureFedCNN
        if model_name == "securefed_cnn":
            if "input_size" not in kwargs:
                # Try to get resize_size from data config
                if "data" in kwargs and "resize_size" in kwargs["data"]:
                    resize_size = kwargs["data"]["resize_size"]
                    print(f"Using resize_size {resize_size} from config as input_size for SecureFedCNN")

                    # Convert list to proper format if needed
                    if isinstance(resize_size, (list, tuple)):
                        input_size = [int(x) for x in resize_size]
                    else:
                        input_size = resize_size

                    # Remove data from kwargs since it's not a model parameter
                    data_config = kwargs.pop("data")

                    # Create the model with explicit input_size
                    return cls._models[model_name](input_size=input_size, **kwargs)
            else:
                print(f"Using provided input_size {kwargs['input_size']} for SecureFedCNN")

        # For other models, create as usual
        return cls._models[model_name](**kwargs)

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model class.

        Args:
            name: Name of the model
            model_class: Model class to register
        """
        cls._models[name] = model_class
