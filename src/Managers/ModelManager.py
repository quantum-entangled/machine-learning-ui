from typing import Any, Protocol

import tensorflow as tf


class Model(Protocol):
    """Protocol for models."""

    name: str | None
    instance: Any
    layers: dict | None


class ModelManager:
    """Manager for operating the model configuration."""

    def __init__(self, model: Model) -> None:
        """Initialize the internal model object."""
        self._model = model

    def create_model(self, model_name: str, output_handler: Any) -> None:
        """Create a model from scratch with the given name."""
        with output_handler:
            output_handler.clear_output(wait=True)

            if not model_name:
                print("Your model name is empty. Please, enter the model name!\u274C")
                return

            if self.is_model(model_name):
                print(
                    "Model with this name already exists. Please, enter a different name!\u274C"
                )
                return

            self._model.instance = tf.keras.Model(name=model_name)
            self._model.name = model_name
            self._model.layers = dict()
            print("You're model is successfully created!\u2705")

    def upload_model(self, file_chooser: Any, output_handler: Any) -> None:
        """Upload a model from the file via the given file chooser."""
        with output_handler:
            output_handler.clear_output(wait=True)

            model_path = self.get_model_path(file_chooser)
            if model_path is None:
                print("Please, select the model first!\u274C")
                return

            self._model.instance = tf.keras.models.load_model(
                filepath=model_path, compile=False
            )
            self._model.name = self._model.instance.name
            self._model.layers = {
                layer.name: layer for layer in self._model.instance.layers
            }
            print("Your model is successfully uploaded!\u2705")

    def get_model_path(self, file_chooser: Any) -> str:
        """Get a model file path via the given file chooser."""
        return file_chooser.selected

    def is_model(self, model_name: str) -> bool:
        """Check if model with the given name exists."""
        return self._model.name == model_name
