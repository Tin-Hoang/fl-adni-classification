"""Model factory for ADNI classification."""

from typing import Dict, Any, Type
import os

from adni_classification.models.base_model import BaseModel
from adni_classification.models.resnet3d import ResNet3D
from adni_classification.models.densenet3d import DenseNet3D
from adni_classification.models.simple_cnn import Simple3DCNN
from adni_classification.models.securefed_cnn import SecureFedCNN
from adni_classification.models.rosanna_cnn import RosannaCNN


class ModelFactory:
    """Factory class for creating model instances."""

    _models: Dict[str, Type[BaseModel]] = {
        "resnet3d": ResNet3D,
        "densenet3d": DenseNet3D,
        "simple3dcnn": Simple3DCNN,
        "securefed_cnn": SecureFedCNN,
        "pretrained_cnn": RosannaCNN,
        "rosanna_cnn": RosannaCNN,
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
            # Extract data config information if needed
            data_config = None
            if "data" in kwargs:
                data_config = kwargs.pop("data")

            # Get classification_mode from data config and pass it to the model
            if data_config and "classification_mode" in data_config:
                classification_mode = data_config["classification_mode"]
                kwargs["classification_mode"] = classification_mode
                print(f"Using classification_mode '{classification_mode}' from data config")
            else:
                classification_mode = kwargs.get("classification_mode", "CN_MCI_AD")

            # If num_classes is explicitly set in the model config, respect that value
            # Otherwise, derive it from the classification mode
            if "num_classes" not in kwargs:
                if classification_mode == "CN_AD":
                    kwargs["num_classes"] = 2
                    print(f"Setting num_classes=2 for classification_mode={classification_mode}")
                else:
                    kwargs["num_classes"] = 3
                    print(f"Setting num_classes=3 for classification_mode={classification_mode}")

            if "input_size" not in kwargs:
                # Try to get resize_size from data config
                if data_config and "resize_size" in data_config:
                    resize_size = data_config["resize_size"]
                    print(f"Using resize_size {resize_size} from config as input_size for SecureFedCNN")

                    # Convert list to proper format if needed
                    if isinstance(resize_size, (list, tuple)):
                        input_size = [int(x) for x in resize_size]
                    else:
                        input_size = resize_size

                    # Add input_size to kwargs
                    kwargs["input_size"] = input_size
            else:
                print(f"Using provided input_size {kwargs['input_size']} for SecureFedCNN")

        # Handle RosannaCNN specific configurations
        elif model_name in ["rosanna_cnn", "pretrained_cnn"]:
            # Extract data config information if needed
            data_config = None
            if "data" in kwargs:
                data_config = kwargs.pop("data")

            # Get classification_mode from data config and set num_classes
            if data_config and "classification_mode" in data_config:
                classification_mode = data_config["classification_mode"]
                if "num_classes" not in kwargs:
                    if classification_mode == "CN_AD":
                        kwargs["num_classes"] = 2
                        print(f"Setting num_classes=2 for classification_mode={classification_mode}")
                    else:
                        kwargs["num_classes"] = 3
                        print(f"Setting num_classes=3 for classification_mode={classification_mode}")

            # Handle pretrained checkpoint parameter
            if "pretrained_checkpoint" in kwargs:
                pretrained_checkpoint = kwargs["pretrained_checkpoint"]
                if pretrained_checkpoint and not os.path.isabs(pretrained_checkpoint):
                    # Make path relative to project root
                    kwargs["pretrained_checkpoint"] = os.path.join(os.getcwd(), pretrained_checkpoint)
                    print(f"Using pretrained checkpoint from: {kwargs['pretrained_checkpoint']}")
                elif pretrained_checkpoint:
                    print(f"Using pretrained checkpoint from: {pretrained_checkpoint}")

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
