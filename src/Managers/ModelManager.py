from typing import Any

import tensorflow as tf
from bqplot import pyplot as bqplt
from IPython.display import display

from DataClasses import Data, Model


class ModelManager:
    """Manager for operating the model configuration."""

    def __init__(self, data: Data, model: Model) -> None:
        """Initialize the internal model object."""
        self._data = data
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
            print("Your model is successfully created!\u2705")

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
        self._model.input_shapes = {
            layer.name: layer.shape[1] for layer in self._model.instance.inputs
        }
        self._model.output_shapes = {
            layer_name: 1 for layer_name in self._model.instance.output_names
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
                        if connect in output_name:
                            mask[i] = True

            if True in mask:
                pop_indices = [i for i, x in enumerate(mask) if x]
                [self._model.instance.outputs.pop(i) for i in reversed(pop_indices)]

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
                filepath=f"../db/Models/{self._model.name}.h5",
                save_format="h5",
                include_optimizer=False,
            )
            output_handler.clear_output()
            print("Your model is successfully saved!\u2705")

    def select_optimizer(self, instance: Any, **kwargs) -> None:
        self._model.optimizer = instance(**kwargs)

    def add_loss(self, layer_name: str, loss: Any) -> None:
        self._model.losses.update({layer_name: loss})

    def add_metric(self, layer_name: str, metric: Any) -> None:
        self._model.metrics.update({layer_name: metric})

    def add_callback(self, instance: Any, **kwargs) -> None:
        self._model.callbacks += [instance(**kwargs)]

    def compile_model(self) -> None:
        self._model.instance.compile(
            optimizer=self._model.optimizer,
            loss=self._model.losses,
            metrics=self._model.metrics,
        )

    def check_shapes(self) -> str:
        for layer_name in list(
            self._data.input_training_columns | self._data.output_training_columns
        ):
            if layer_name in self._model.input_shapes:
                shape = self._model.input_shapes[layer_name]
            else:
                shape = self._model.output_shapes[layer_name]

            if self._data.num_columns_per_layer[layer_name] < shape:
                return layer_name

        return str()

    def fit_model(
        self, batch_size: int, num_epochs: int, validation_split: float
    ) -> None:
        history = self._model.instance.fit(
            x={
                layer_name: self._data.file[
                    :, self._data.input_training_columns[layer_name]
                ]
                for layer_name in self._data.input_training_columns.keys()
            },
            y={
                layer_name: self._data.file[
                    :, self._data.output_training_columns[layer_name]
                ]
                for layer_name in self._data.output_training_columns.keys()
            },
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=validation_split,
            callbacks=self._model.callbacks,
            verbose=1,
        )

        self._model.training_history = history.history

    def plot_history(self, y: Any, color: Any, same_figure: bool) -> None:
        y_data = self._model.training_history[y]
        x_data = [i + 1 for i, _ in enumerate(y_data)]

        if same_figure:
            fig = bqplt.current_figure()
        else:
            fig = bqplt.figure()

        fig.min_aspect_ratio = 1
        fig.max_aspect_ratio = 1
        fig.fig_margin = {"top": 5, "bottom": 35, "left": 40, "right": 5}

        bqplt.plot(x=x_data, y=y_data, colors=[color], labels=[y], figure=fig)
        bqplt.xlabel("Epoch")
        bqplt.xlim(min=min(x_data) - 1, max=max(x_data) + 1)
        bqplt.legend()
        bqplt.show()

    @property
    def model(self) -> Model:
        return self._model
