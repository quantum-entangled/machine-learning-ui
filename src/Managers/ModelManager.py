from typing import Any

import tensorflow as tf
from bqplot import pyplot as bqplt
from IPython.display import display

from DataClasses import Data, Model
from Enums.WatchTypes import Watch


class ModelManager:
    """Manager for operating the model configuration."""

    def __init__(self, data: Data, model: Model) -> None:
        """Initialize model object."""
        self._data = data
        self._model = model
        self._watchers = list()

    def create_model(self, model_name: str) -> None:
        """Create model from scratch."""
        self._model.instance = tf.keras.Model(
            inputs=list(), outputs=list(), name=model_name
        )

        self.refresh_model()
        self.callback_watchers(callback_type=Watch.MODEL)

    def upload_model(self, model_path: Any) -> None:
        """Upload TensorFlow model."""
        self._model.instance = tf.keras.models.load_model(
            filepath=model_path, compile=False
        )

        self.refresh_model()
        self.callback_watchers(callback_type=Watch.MODEL)

    def refresh_model(self) -> None:
        self._model.name = self._model.instance.name
        self._model.input_layers = {
            name: layer
            for name, layer in zip(
                self._model.instance.input_names, self._model.instance.inputs
            )
        }
        self._model.output_layers = {
            name: layer
            for name, layer in zip(
                self._model.instance.output_names, self._model.instance.outputs
            )
        }
        self._model.layers = self._model.input_layers | self._model.output_layers
        self._model.input_shapes = {
            layer.name: layer.shape[1] for layer in self._model.instance.inputs
        }
        self._model.output_shapes = {
            layer_name: 1 for layer_name in self._model.instance.output_names
        }
        self._model.input_model_columns = {
            name: list() for name in self._model.input_layers
        }
        self._model.output_model_columns = {
            name: list() for name in self._model.output_layers
        }
        self._model.layers_fullness = {
            name: 0 for name in self._model.input_layers | self._model.output_layers
        }

    def add_layer(self, layer_instance: Any, connect_to: Any, **kwargs) -> None:
        if not connect_to:
            layer = {kwargs["name"]: layer_instance(**kwargs)}

            self._model.input_layers.update(layer)
        else:
            if isinstance(connect_to, str):
                connect = self._model.layers[connect_to]
            else:
                connect = [self._model.layers[name] for name in connect_to]

            layer = {kwargs["name"]: layer_instance(**kwargs)(connect)}

        self._model.layers.update(layer)
        self.callback_watchers(callback_type=Watch.LAYER_ADDED)

    def set_model_outputs(self, outputs_names: Any) -> None:
        self._model.output_layers = {
            name: self._model.layers[name] for name in outputs_names
        }
        self._model.instance = tf.keras.Model(
            inputs=self._model.input_layers,
            outputs=self._model.output_layers,
            name=self._model.name,
        )

        self.refresh_model()
        self.callback_watchers(callback_type=Watch.OUTPUTS_SET)

    def show_model_summary(self, output_handler: Any) -> None:
        output_handler.clear_output(wait=True)

        if not self._model.instance:
            return

        if not self._model.layers:
            return

        with output_handler:
            print("\n")
            display(self._model.instance.summary())

    def plot_model(self) -> None:
        """Plot TensorFlow model graph."""
        display(
            tf.keras.utils.plot_model(
                self._model.instance,
                to_file=f"../db/Images/{self._model.name}.png",
                show_shapes=True,
                rankdir="LR",
                dpi=200,
            )
        )

    def save_model(self) -> None:
        "Save model to '.h5' format."
        self._model.instance.save(
            filepath=f"../db/Models/{self._model.name}.h5",
            save_format="h5",
        )

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

    def add_columns(self, layer_type: str, layer_name: str, columns: Any) -> None:
        if layer_type == "input":
            self._model.input_model_columns[layer_name].extend(columns)
        else:
            self._model.output_model_columns[layer_name].extend(columns)

        self._model.layers_fullness[layer_name] += len(columns)

    def check_layer_capacity(
        self, layer_type: str, layer_name: str, num_columns: int
    ) -> bool:
        if layer_type == "input":
            shape = self._model.input_shapes[layer_name]
        else:
            shape = self._model.output_shapes[layer_name]

        current_num_columns = self._model.layers_fullness[layer_name]

        return False if num_columns + current_num_columns > shape else True

    def callback_watchers(self, callback_type: str) -> None:
        for watcher in self._watchers:
            callback = getattr(watcher, callback_type, None)

            if callable(callback):
                callback()

    def model_exists(self) -> bool:
        return True if self._model.instance else False

    @property
    def model(self) -> Model:
        return self._model

    @property
    def name(self) -> str:
        return self._model.name

    @property
    def input_layers(self) -> dict[str, Any]:
        return self._model.input_layers

    @property
    def output_layers(self) -> dict[str, Any]:
        return self._model.output_layers

    @property
    def layers(self) -> dict[str, Any]:
        return self._model.layers

    @property
    def input_shapes(self) -> dict[str, int]:
        return self._model.input_shapes

    @property
    def output_shapes(self) -> dict[str, int]:
        return self._model.output_shapes

    @property
    def input_model_columns(self) -> dict[str, list[str]]:
        return self._model.input_model_columns

    @property
    def output_model_columns(self) -> dict[str, list[str]]:
        return self._model.output_model_columns

    @property
    def layers_fullness(self) -> dict[str, int]:
        return self._model.layers_fullness

    @property
    def watchers(self) -> list[Any]:
        return self._watchers

    @watchers.setter
    def watchers(self, watchers_list: list[Any]) -> None:
        self._watchers = watchers_list
