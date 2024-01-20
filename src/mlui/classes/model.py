import io
import tempfile
import typing

import altair as alt
import pandas as pd
import tensorflow as tf

import mlui.classes.errors as errors
import mlui.enums as enums
import mlui.tools as tools
import mlui.types.classes as t


class Model:
    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        self._object: t.Object = tf.keras.Model(inputs=list(), outputs=list())
        self._built: bool = False
        self._set_config()
        self.update_state()

    def _set_config(self) -> None:
        self._name: str = self._object.name
        self._inputs: t.Layers = typing.cast(t.Layers, self._object.input_names)
        self._outputs: t.Layers = typing.cast(t.Layers, self._object.output_names)
        self._input_shape: t.LayerShape = self._get_processed_shape("input")
        self._output_shape: t.LayerShape = self._get_processed_shape("output")
        self._optimizer: t.Optimizer = self._object.optimizer
        self._losses: t.LayerLosses = self._get_processed_losses()
        self._metrics: t.LayerMetrics = dict.fromkeys(self._outputs, list())
        self._callbacks: t.Callbacks = dict()
        self._compiled: bool = self._object._is_compiled
        self._history: t.DataFrame = pd.DataFrame()

    def update_state(self) -> None:
        self._input_features: t.LayerFeatures = dict.fromkeys(self._inputs, list())
        self._output_features: t.LayerFeatures = dict.fromkeys(self._outputs, list())
        self._input_configured: t.LayerConfigured = dict.fromkeys(self._inputs, False)
        self._output_configured: t.LayerConfigured = dict.fromkeys(self._outputs, False)

    def _shapes_to_list(self, shapes: t.Shapes) -> list[t.Shape]:
        if isinstance(shapes, dict):
            return list(shapes.values())

        return [shapes] if isinstance(shapes, tuple) else shapes

    def _get_processed_shape(self, at: t.Side) -> t.LayerShape:
        if at == "input":
            layers = self._inputs
            shapes = self._object.input_shape
        else:
            layers = self._outputs
            shapes = self._object.output_shape

        shapes = self._shapes_to_list(shapes)

        return {layer: shape[1] for layer, shape in zip(layers, shapes)}

    def _get_processed_losses(self) -> t.LayerLosses:
        try:
            losses = typing.cast(list, self._object.loss)

            return {layer: loss for layer, loss in zip(self._outputs, losses)}
        except AttributeError:
            return dict.fromkeys(self._outputs)

    def _get_processed_data(self, data: t.DataFrame, at: t.Side) -> t.LayerData:
        if at == "input":
            layers = self._inputs
            features = self._input_features
        else:
            layers = self._outputs
            features = self._output_features

        return {layer: data[features[layer]].to_numpy() for layer in layers}

    def set_optimizer(self, entity: str, params: t.OptimizerParams) -> None:
        try:
            prototype = enums.optimizers.classes[entity]
            self._optimizer = prototype(**params)
        except KeyError:
            raise errors.SetError("There is no prototype for this optimizer!")
        except (ValueError, AttributeError, TypeError):
            raise errors.SetError("Unable to set the optimizer!")

    def get_optimizer(self) -> t.Optimizer:
        return self._optimizer.name if self._optimizer else None

    def set_loss(self, layer: str, entity: str) -> None:
        self._losses[layer] = entity

    def get_loss(self, layer: str) -> t.Loss:
        return self._losses[layer] if self._losses.get(layer) else None

    def set_metrics(self, layer: str, entities: list[str]) -> None:
        self._metrics[layer] = entities

    def get_metrics(self, layer: str) -> t.Metrics:
        return self._metrics[layer].copy() if self._metrics.get(layer) else list()

    def compile(self) -> None:
        if not self._optimizer_is_set:
            raise errors.ModelError("Please, set the model optimizer!")

        if not self._losses_are_set:
            raise errors.ModelError(
                "Please, set the loss function for each output layer!"
            )

        try:
            self._object.compile(
                optimizer=self._optimizer, loss=self._losses, metrics=self._metrics
            )
        except (ValueError, AttributeError, TypeError):
            raise errors.ModelError("Unable to compile the model!")

        self._compiled = True

    def set_features(self, layer: str, columns: t.Columns, at: t.Side) -> None:
        if not columns:
            raise errors.SetError("Please, select at least one column!")

        if at == "input":
            shape = self._input_shape.get(layer)
            configured = self._input_configured
            features = self._input_features
        else:
            shape = self._output_shape.get(layer)
            configured = self._output_configured
            features = self._output_features

        if not shape:
            raise errors.SetError("There is no such layer in the model!")

        if len(columns) > shape:
            raise errors.SetError("Please, select fewer columns!")

        if len(columns) == shape:
            configured[layer] = True
        else:
            configured[layer] = False

        features[layer] = columns

    def get_features(self, layer: str, at: t.Side) -> t.Features:
        if at == "input":
            features = self._input_features
        else:
            features = self._output_features

        return features[layer].copy() if features.get(layer) else list()

    def set_callback(self, entity: str, params: t.CallbackParams) -> None:
        try:
            prototype = enums.callbacks.classes[entity]
            self._callbacks[entity] = prototype(**params)
        except KeyError:
            raise errors.SetError("There is no prototype for this callback!")
        except (ValueError, AttributeError, TypeError):
            raise errors.SetError("Unable to set the callback!")

    def get_callback(self, entity: str) -> t.Callback:
        return self._callbacks.get(entity)

    def delete_callback(self, entity: str) -> None:
        self._callbacks.pop(entity, None)

    def fit(
        self, data: t.DataFrame, batch_size: int, num_epochs: int, val_split: float
    ) -> None:
        if tools.data.contains_nonnumeric_dtypes(data):
            raise errors.ModelError("The data for fitting contains non-numeric values!")

        try:
            logs = self._object.fit(
                x=self._get_processed_data(data, "input"),
                y=self._get_processed_data(data, "output"),
                batch_size=batch_size,
                epochs=num_epochs,
                validation_split=val_split,
                callbacks=self._callbacks.values(),
            )
        except (RuntimeError, ValueError, AttributeError, TypeError):
            raise errors.ModelError("Unable to fit the model!")

        self._update_history(pd.DataFrame(logs.history))

    def _update_history(self, logs: t.DataFrame) -> None:
        history_len = len(self._history)
        logs.insert(0, "epoch", range(history_len + 1, history_len + len(logs) + 1))

        self._history = pd.concat([self._history, logs])

    def plot_history(self, y: t.LogsNames, points: bool) -> t.Chart:
        if not y:
            raise errors.PlotError("Please, select at least one log!")

        try:
            logs = self._history.loc[:, ["epoch", *y]]
            melted_logs = logs.melt(
                "epoch", var_name="log_name", value_name="log_value"
            )
            chart = (
                alt.Chart(melted_logs)
                .mark_line(point=points)
                .encode(
                    x=alt.X("epoch").scale(zero=False).title("Epoch"),
                    y=alt.Y("log_value").scale(zero=False).title("Value"),
                    color=alt.Color("log_name")
                    .scale(scheme="set1")
                    .legend(title="Log"),
                )
                .interactive(bind_x=True, bind_y=True)
                .properties(height=500)
            )
        except (ValueError, AttributeError, TypeError):
            raise errors.PlotError("Unable to display the plot!")

        return chart

    @property
    def name(self) -> str:
        return self._name

    @property
    def inputs(self) -> t.Layers:
        return self._inputs.copy()

    @property
    def outputs(self) -> t.Layers:
        return self._outputs.copy()

    @property
    def input_shape(self) -> t.LayerShape:
        return self._input_shape.copy()

    @property
    def output_shape(self) -> t.LayerShape:
        return self._output_shape.copy()

    @property
    def input_configured(self) -> bool:
        return (
            True
            if self._input_configured and all(self._input_configured.values())
            else False
        )

    @property
    def output_configured(self) -> bool:
        return (
            True
            if self._output_configured and all(self._output_configured.values())
            else False
        )

    @property
    def _optimizer_is_set(self) -> bool:
        return True if self._optimizer else False

    @property
    def _losses_are_set(self) -> bool:
        return True if self._losses and all(self._losses.values()) else False

    @property
    def built(self) -> bool:
        return self._built

    @property
    def compiled(self) -> bool:
        return self._compiled

    @property
    def history(self) -> t.DataFrame:
        return self._history.copy()

    @property
    def summary(self) -> None:
        self._object.summary()

    @property
    def graph(self) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tf.keras.utils.plot_model(
                self._object, to_file=tmp.name, show_shapes=True, rankdir="LR", dpi=200
            )

            graph = tmp.read()

        return graph

    @property
    def as_bytes(self) -> bytes:
        with tempfile.NamedTemporaryFile() as tmp:
            self._object.save(filepath=tmp.name, save_format="h5")

            model_as_bytes = tmp.read()

        return model_as_bytes


class UploadedModel(Model):
    def __init__(self) -> None:
        super().__init__()

    def upload(self, buff: io.BytesIO) -> None:
        """Upload a TensorFlow model.

        Parameters
        ----------
        buff : File-like object
            Buffer object to upload.

        Raises
        ------
        errors.UploadError
            For errors during the uploading procedure.
        """
        try:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(buff.getbuffer())

                model = typing.cast(t.Object, tf.keras.models.load_model(tmp.name))
                tools.model.validate_shapes(model.input_shape)
                tools.model.validate_shapes(model.output_shape)

                self._object = model
                self._built = True
                self._set_config()
                self.update_state()
        except (ValueError, errors.ValidateModelError) as error:
            raise errors.UploadError(error)

    def evaluate(self, data: t.DataFrame, batch_size: int) -> t.EvaluationResults:
        if tools.data.contains_nonnumeric_dtypes(data):
            raise errors.ModelError(
                "The data for evaluation contains non-numeric values!"
            )

        try:
            logs = typing.cast(
                dict[str, float],
                self._object.evaluate(
                    x=self._get_processed_data(data, "input"),
                    y=self._get_processed_data(data, "output"),
                    batch_size=batch_size,
                    callbacks=self._callbacks.values(),
                    return_dict=True,
                ),
            )  # Type-cast the return value as 'return_dict' is set to True
            results = pd.DataFrame(logs.items(), columns=["Name", "Value"])
        except (RuntimeError, ValueError, AttributeError, TypeError):
            raise errors.ModelError("Unable to evaluate the model!")

        return results

    def predict(self, data: t.DataFrame, batch_size: int) -> t.Predictions:
        if tools.data.contains_nonnumeric_dtypes(data):
            raise errors.ModelError(
                "The data for predictions contains non-numeric values!"
            )

        try:
            arrays = self._object.predict(
                x=self._get_processed_data(data, "input"),
                batch_size=batch_size,
                callbacks=self._callbacks.values(),
            )

            if isinstance(arrays, dict):
                predictions = [pd.DataFrame(array) for array in arrays.values()]
            elif isinstance(arrays, list):
                predictions = [pd.DataFrame(array) for array in arrays]
            else:
                predictions = [pd.DataFrame(arrays)]
        except (RuntimeError, ValueError, AttributeError, TypeError):
            raise errors.ModelError("Unable to make the prediction!")

        return predictions


class CreatedModel(Model):
    def __init__(self) -> None:
        self.reset_state()

    def reset_state(self) -> None:
        super().reset_state()

        self._layers: t.LayerObject = dict()

    def set_name(self, name: str) -> None:
        self._name = name

    def set_layer(
        self,
        entity: str,
        name: str,
        params: t.LayerParams,
        connection: t.LayerConnection,
    ) -> None:
        if name in self._layers:
            raise errors.SetError("Layer with this name already exists!")

        try:
            prototype = enums.layers.classes[entity]

            if connection is None:
                layer = prototype(name=name, **params)
            else:
                layer = prototype(name=name, **params)(connection)
        except KeyError:
            raise errors.SetError("There is no prototype for this layer!")
        except (ValueError, AttributeError, TypeError):
            raise errors.SetError("Unable to set the layer!")

        if entity == "Input":
            self._inputs.append(name)

        self._layers[name] = layer

    def delete_last_layer(self) -> None:
        try:
            name = self._layers.popitem()[0]
        except KeyError:
            raise errors.DeleteError("There are no layers to remove!")

        try:
            self._inputs.remove(name)
        except ValueError:
            pass

        try:
            self._outputs.remove(name)
        except ValueError:
            pass

    def set_outputs(self, outputs: t.Layers) -> None:
        if not outputs:
            raise errors.SetError("Please, select at least one output!")

        self._outputs = outputs

    def create(self) -> None:
        input_layers = {name: self._layers[name] for name in self._inputs}
        output_layers = {name: self._layers[name] for name in self._outputs}

        if not input_layers or not output_layers:
            raise errors.CreateError("There are no input or output layers!")

        try:
            self._object = tf.keras.Model(
                inputs=input_layers, outputs=output_layers, name=self._name
            )
        except (ValueError, AttributeError, TypeError):
            raise errors.CreateError("Unable to create the model!")

        self._built = True
        self._set_config()
        self.update_state()

    @property
    def layers(self) -> t.LayerObject:
        return self._layers.copy()
