from typing import Any, Dict, Protocol

import tensorflow as tf


class Model(Protocol):
    """Protocol for models."""

    name: str | None
    instance: Any
    layers: Dict[str, Any]


class ModelManager:
    """Manager for operating the model configuration."""

    def __init__(self, model: Model) -> None:
        """Initialize the internal model object."""
        self._model = model

    def create_model(self, model_name: str, output_handler: Any) -> None:
        """Create a model from scratch with the given name."""
        output_handler.clear_output(wait=True)

        if not model_name:
            with output_handler:
                print("Your model name is empty. Please, enter the model name!\u274C")
            return

        if self._model.name == model_name:
            with output_handler:
                print(
                    "Model with this name already exists. Please, enter a different name!\u274C"
                )
            return

        self._model.instance = tf.keras.Model(
            inputs=list(), outputs=list(), name=model_name
        )
        self._model.name = model_name
        self._model.layers = dict()

        with output_handler:
            print("You're model is successfully created!\u2705")

    def upload_model(self, file_chooser: Any, output_handler: Any) -> None:
        """Upload a model from the file via the given file chooser."""
        output_handler.clear_output(wait=True)

        model_path = self.get_model_path(file_chooser)
        if model_path is None:
            with output_handler:
                print("Please, select the model first!\u274C")
            return

        self._model.instance = tf.keras.models.load_model(
            filepath=model_path, compile=False
        )
        self._model.name = self._model.instance.name
        self._model.layers = {
            layer.name: layer for layer in self._model.instance.layers
        }

        with output_handler:
            print("Your model is successfully uploaded!\u2705")

    def get_model_path(self, file_chooser: Any) -> str:
        """Get a model file path via the given file chooser."""
        return file_chooser.selected

    def add_layer(
        self,
        layer_type: Any,
        instance: Any,
        connect_to: Any,
        output_handler: Any,
        **kwargs,
    ) -> None:
        output_handler.clear_output(wait=True)

        layer_name = kwargs["name"]

        if not self._model.instance:
            with output_handler:
                print("Please, create or upload the model first!\u274C")
            return

        if not layer_name:
            with output_handler:
                print("Please, enter the layer name first!\u274C")
            return

        if layer_name in self._model.layers:
            with output_handler:
                print(
                    "Layer with this name already exists. Please, enter a different name!\u274C"
                )
            return

        if not connect_to and not layer_type == "Input":
            with output_handler:
                print("Please, choose the connection layer first!\u274C")
            return

        if connect_to and layer_type == "Input":
            with output_handler:
                print(
                    "You can't connect Input layer to the other ones. Please, choose another layer type or unset the connection!\u274C"
                )
            return

        if not connect_to and layer_type == "Input":
            self._model.layers.update({layer_name: instance(**kwargs)})
            self.update_model(layer_type=layer_type, layer_name=layer_name)

            with output_handler:
                print(f"Layer '{layer_name}' is successfully added!\u2705")
            return

        connect_to_layer = self._model.layers[connect_to]
        self._model.layers.update({layer_name: instance(**kwargs)(connect_to_layer)})
        self.update_model(layer_type=layer_type, layer_name=layer_name)

        with output_handler:
            print(f"Layer '{layer_name}' is successfully added!\u2705")

    def update_model(self, layer_type: Any, layer_name: Any) -> None:
        layer = self._model.layers[layer_name]

        if layer_type == "Input":
            self._model.instance = tf.keras.Model(
                inputs=[*self._model.instance.inputs, layer],
                outputs=self._model.instance.outputs,
                name=self._model.name,
            )
        else:
            self._model.instance = tf.keras.Model(
                inputs=self._model.instance.inputs,
                outputs=[layer],
                name=self._model.name,
            )

    @property
    def model(self) -> Model:
        return self._model
