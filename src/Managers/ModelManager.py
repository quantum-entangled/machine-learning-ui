from typing import Any, Protocol

import tensorflow as tf
from IPython.display import display


class Model(Protocol):
    """Protocol for models."""

    name: str | None
    instance: Any
    layers: dict[Any, Any]
    layers_shapes: dict[Any, Any]
    input_names: list[str]
    output_names: list[str]


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

        model_path = self.get_model_path(file_chooser=file_chooser)

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
        self._model.layers_shapes = {
            layer.name: (
                layer.output_shape[0][1]
                if isinstance(layer.output_shape, list)
                else layer.output_shape[1]
            )
            for layer in self._model.instance.layers
        }
        self._model.input_names = self._model.instance.input_names
        self._model.output_names = self._model.instance.output_names

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

        if not connect_to and not self._model.layers and layer_type != "Input":
            with output_handler:
                print("Please, add some layers first!\u274C")
            return

        if not connect_to:
            self._model.layers.update({layer_name: instance(**kwargs)})
            self.update_model(
                layer_type=layer_type, layer_name=layer_name, connect_to=connect_to
            )

            with output_handler:
                print(f"Layer '{layer_name}' is successfully added!\u2705")
            return

        if isinstance(connect_to, str):
            connection = self._model.layers[connect_to]
        else:
            connection = [self._model.layers[connect] for connect in connect_to]

        self._model.layers.update({layer_name: instance(**kwargs)(connection)})
        self.update_model(
            layer_type=layer_type, layer_name=layer_name, connect_to=connect_to
        )

        with output_handler:
            print(f"Layer '{layer_name}' is successfully added!\u2705")

    def update_model(self, layer_type: Any, layer_name: Any, connect_to: Any) -> None:
        layer = self._model.layers[layer_name]

        if layer_type == "Input":
            self._model.instance = tf.keras.Model(
                inputs=[*self._model.instance.inputs, layer],
                outputs=self._model.instance.outputs,
                name=self._model.name,
            )
        else:
            output_names = [output.name for output in self._model.instance.outputs]

            if isinstance(connect_to, str):
                mask = [connect_to in output_name for output_name in output_names]
            else:
                mask = [False for _ in range(len(output_names))]

                for connect in connect_to:
                    for i, output_name in enumerate(output_names):
                        if connect == output_name:
                            mask[i] = True

            if True in mask:
                pop_indices = [i for i, x in enumerate(mask) if x]
                [self._model.instance.outputs.pop(i) for i in pop_indices]

            self._model.instance = tf.keras.Model(
                inputs=self._model.instance.inputs,
                outputs=[*self._model.instance.outputs, layer],
                name=self._model.name,
            )

    def show_model_summary(self, output_handler: Any) -> None:
        output_handler.clear_output(wait=True)

        if not self._model.instance:
            return

        if not self._model.layers:
            return

        with output_handler:
            print("\n")
            display(self._model.instance.summary())

    def plot_model(self, output_handler: Any) -> None:
        output_handler.clear_output(wait=True)

        if not self._model.instance:
            with output_handler:
                print("Please, create or upload the model first!\u274C")
            return

        if not self._model.layers:
            with output_handler:
                print("Please, add some layers first!\u274C")
            return

        with output_handler:
            display(
                tf.keras.utils.plot_model(
                    self._model.instance,
                    to_file=f"../db/Images/{self._model.name}.png",
                    show_shapes=True,
                    dpi=250,
                )
            )

    def save_model(self, output_handler: Any) -> None:
        output_handler.clear_output()

        if not self._model.instance:
            with output_handler:
                print("Please, create or upload the model first!\u274C")
            return

        with output_handler:
            self._model.instance.save(
                filepath=f"../db/Models/save/{self._model.name}.h5",
                save_format="h5",
                include_optimizer=False,
            )
            output_handler.clear_output()
            print("Your model is successfully saved!\u2705")

    @property
    def model(self) -> Model:
        return self._model
