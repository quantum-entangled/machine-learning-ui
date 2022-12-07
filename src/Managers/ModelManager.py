from typing import Any

import tensorflow as tf
from bqplot import pyplot as bqplt
from IPython.display import display

from DataClasses import Data, Model
from Enums.ObserveTypes import Observe


class ModelManager:
    """Manager for operating the model configuration."""

    def __init__(self, data: Data, model: Model) -> None:
        """Initialize model object."""
        self._data = data
        self._model = model
        self._observers = list()

    def create_model(self, model_name: str) -> None:
        """Create model from scratch."""
        self._model.instance = tf.keras.Model(
            inputs=list(), outputs=list(), name=model_name
        )
        self.refresh_model()
        self.notify_observers(callback_type=Observe.MODEL)

    def upload_model(self, model_path: Any) -> None:
        """Upload TensorFlow model."""
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        self._model.instance = tf.keras.models.load_model(filepath=model_path)
        self.refresh_model()
        self.notify_observers(callback_type=Observe.MODEL)

    def refresh_model(self) -> None:
        """Refresh model instance and its properties."""
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
        self._model.losses = {name: list() for name in self._model.output_layers}
        self._model.metrics = {name: list() for name in self._model.output_layers}

    def add_layer(self, layer_instance: Any, connect_to: Any, **kwargs) -> None:
        """Add layers to model."""
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
        self.notify_observers(callback_type=Observe.LAYER_ADDED)

    def set_model_outputs(self, outputs_names: Any) -> None:
        """Set model outputs."""
        self._model.output_layers = {
            name: self._model.layers[name] for name in outputs_names
        }
        self._model.instance = tf.keras.Model(
            inputs=self._model.input_layers,
            outputs=self._model.output_layers,
            name=self._model.name,
        )

        self.refresh_model()
        self.notify_observers(callback_type=Observe.OUTPUTS_SET)

    def show_model_summary(self) -> None:
        """Show model summary."""
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
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        self._model.instance.save(
            filepath=f"../db/Models/{self._model.name}.h5",
            save_format="h5",
        )

    def select_optimizer(self, optimizer: Any, **kwargs) -> None:
        """Instantiate model optimizer."""
        self._model.optimizer = optimizer(**kwargs)
        self.notify_observers(callback_type=Observe.OPTIMIZER_SELECTED)

    def add_loss(self, layer: str, loss: Any) -> None:
        """Instantiate model losses."""
        self._model.losses[layer] = loss()
        self.notify_observers(callback_type=Observe.LOSSES_SELECTED)

    def add_metric(self, layer: str, metric: Any) -> None:
        """Instantiate model metrics."""
        self._model.metrics[layer].append(metric())

    def add_callback(self, callback: Any, **kwargs) -> None:
        """Instantiate model callbacks."""
        self._model.callbacks.append(callback(**kwargs))

    def compile_model(self) -> None:
        """Compile model."""
        self._model.instance.compile(
            optimizer=self._model.optimizer,
            loss=self._model.losses,
            metrics=self._model.metrics,
        )
        self._model.compiled = True
        self.notify_observers(callback_type=Observe.MODEL_COMPILED)

    def fit_model(self, batch_size: int, num_epochs: int, val_split: float) -> None:
        """Fit model."""
        history = self._model.instance.fit(
            x=self._data.input_train_data,
            y=self._data.output_train_data,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=val_split,
            callbacks=self._model.callbacks,
        )

        self._model.training_history = history.history
        self.notify_observers(callback_type=Observe.MODEL_TRAINED)

    def evaluate_model(self, batch_size: int) -> None:
        """Evaluate model."""
        results = self._model.instance.evaluate(
            x=self._data.input_test_data,
            y=self._data.output_test_data,
            batch_size=batch_size,
            callbacks=self._model.callbacks,
            return_dict=True,
            verbose=0,
        )

        for name, value in results.items():
            print(f"{name}: {value}")

    def make_predictions(self, batch_size: int) -> None:
        """Make model predictions."""
        predictions = self._model.instance.predict(
            x={
                name: self._data.file[values]
                for name, values in self._data.input_columns.items()
            },
            batch_size=batch_size,
            callbacks=self._model.callbacks,
            verbose=0,
        )

        if isinstance(predictions, dict):
            for name, value in predictions.items():
                print(f"{name}: {list(value.flatten())}", end="\n\n")
        else:
            print(f"{self._model.output_layers.keys()}: {list(predictions.flatten())}")

    def plot_history(self, y: Any, color: Any, same_figure: bool) -> None:
        """Plot training history."""
        y_data = self._model.training_history[y]
        x_data = [i + 1 for i, _ in enumerate(y_data)]

        if same_figure:
            fig = bqplt.current_figure()
        else:
            fig = bqplt.figure()

        fig.min_aspect_ratio = 1
        fig.max_aspect_ratio = 1
        fig.fig_margin = {"top": 5, "bottom": 35, "left": 45, "right": 5}
        fig.legend_style = {"stroke-width": 1}

        bqplt.plot(x=x_data, y=y_data, colors=[color], labels=[y], figure=fig)
        bqplt.xlabel("Epoch")
        bqplt.xlim(min=min(x_data) - 1, max=max(x_data) + 1)
        bqplt.legend()
        bqplt.show()

    def check_layer_capacity(
        self, layer_type: str, layer: str, num_columns: int
    ) -> bool:
        """Check if layer is overfilled."""
        if layer_type == "input":
            shape = self._model.input_shapes[layer]
        else:
            shape = self._model.output_shapes[layer]

        current_num_columns = self._data.columns_per_layer[layer]

        return False if num_columns + current_num_columns > shape else True

    def notify_observers(self, callback_type: str) -> None:
        """Notifier for manager's observers."""
        for observer in self._observers:
            callback = getattr(observer, callback_type, None)

            if callable(callback):
                callback()

    def model_exists(self) -> bool:
        """Check if model instance exists."""
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
    def compiled(self) -> bool:
        return self._model.compiled

    @property
    def optimizer(self) -> Any:
        return self._model.optimizer

    @property
    def losses(self) -> dict[str, Any]:
        return self._model.losses

    @property
    def metrics(self) -> dict[str, list[Any]]:
        return self._model.metrics

    @property
    def callbacks(self) -> list[Any]:
        return self._model.callbacks

    @property
    def training_history(self) -> dict[str, Any]:
        return self._model.training_history

    @property
    def observers(self) -> list[Any]:
        return self._observers

    @observers.setter
    def observers(self, observers_list: list[Any]) -> None:
        self._observers = observers_list
