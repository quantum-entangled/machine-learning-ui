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
    """
    Class representing a machine learning model.

    This class provides methods for managing and interacting with a TensorFlow machine
    learning model.
    """

    def __init__(self) -> None:
        """Initialize an empty model."""
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset the model state.

        This method resets the internal state of the model, including its configuration
        and assigned features.
        """
        self._object: t.Object = tf.keras.Model(inputs=list(), outputs=list())
        self._built: bool = False
        self._set_config()
        self.update_state()

    def _set_config(self) -> None:
        """Set the configuration attributes for the model."""
        self._name: str = self._object.name
        self._inputs: t.Layers = typing.cast(t.Layers, self._object.input_names)
        self._outputs: t.Layers = typing.cast(t.Layers, self._object.output_names)
        self._input_shape: t.LayerShape = self._get_processed_shape("input")
        self._output_shape: t.LayerShape = self._get_processed_shape("output")
        self._optimizer: t.Optimizer = None
        self._losses: t.LayerLosses = dict.fromkeys(self._outputs)
        self._metrics: t.LayerMetrics = dict.fromkeys(self._outputs, list())
        self._callbacks: t.Callbacks = dict()
        self._compiled: bool = self._object._is_compiled
        self._history: t.DataFrame = pd.DataFrame()

    def update_state(self) -> None:
        """
        Update the internal state of the model, resetting the input and output features.
        """
        self._input_features: t.LayerFeatures = dict.fromkeys(self._inputs, list())
        self._output_features: t.LayerFeatures = dict.fromkeys(self._outputs, list())
        self._input_configured: t.LayerConfigured = dict.fromkeys(self._inputs, False)
        self._output_configured: t.LayerConfigured = dict.fromkeys(self._outputs, False)

    def _shapes_to_list(self, shapes: t.Shapes) -> list[t.Shape]:
        """
        Convert shapes from a dictionary or a single tuple to a list.

        Parameters
        ----------
        shapes : dict of tuples, list of tuples or tuple
            Input or output shapes. Single shape is a `tuple` of `(None, int)`.

        Returns
        -------
        list of tuples
            Converted shapes.
        """
        if isinstance(shapes, dict):
            return list(shapes.values())

        return [shapes] if isinstance(shapes, tuple) else shapes

    def _get_processed_shape(self, at: t.Side) -> t.LayerShape:
        """
        Retrieve and process the shapes of input or output layers.

        Parameters
        ----------
        at : {'input', 'output'}
            Side to retrieve shapes for.

        Returns
        -------
        dict of {str to int}
            Processed shapes for the specified side.
        """
        if at == "input":
            layers = self._inputs
            shapes = self._object.input_shape
        else:
            layers = self._outputs
            shapes = self._object.output_shape

        shapes = self._shapes_to_list(shapes)

        return {layer: shape[1] for layer, shape in zip(layers, shapes)}

    def _get_processed_data(self, data: t.DataFrame, at: t.Side) -> t.LayerData:
        """
        Process the input or output data based on the specified side.

        Parameters
        ----------
        data : DataFrame
            Input or output data.
        at : {'input', 'output'}
            Side to process data for.

        Returns
        -------
        dict of {str to NDArray}
            Processed data.
        """
        if at == "input":
            layers = self._inputs
            features = self._input_features
        else:
            layers = self._outputs
            features = self._output_features

        return {layer: data[features[layer]].to_numpy() for layer in layers}

    def set_optimizer(self, entity: str, params: t.OptimizerParams) -> None:
        """
        Set the optimizer for the model.

        Parameters
        ----------
        entity : str
            Name of the optimizer type.
        params : OptimizerParams
            Parameters for the optimizer.

        Raises
        ------
        SetError
            If there is an issue setting the optimizer.
        """
        try:
            prototype = enums.optimizers.classes[entity]
            self._optimizer = prototype(**params)
        except KeyError:
            raise errors.SetError("There is no prototype for this optimizer!")
        except (ValueError, AttributeError, TypeError):
            raise errors.SetError("Unable to set the optimizer!")

    def get_optimizer(self) -> str | None:
        """
        Get the name of the current optimizer.

        Returns
        -------
        str or None
            Name of the optimizer.
        """
        return typing.cast(str, self._optimizer.name) if self._optimizer else None

    def set_loss(self, layer: str, entity: str) -> None:
        """
        Set the loss function for a specific output layer.

        Parameters
        ----------
        layer : str
            Name of the output layer.
        entity : str
            Name of the loss function type.
        """
        self._losses[layer] = entity

    def get_loss(self, layer: str) -> t.Loss:
        """
        Get the loss function for a specific output layer.

        Parameters
        ----------
        layer : str
            Name of the output layer.

        Returns
        -------
        str or None
            Name of the loss function.
        """
        return self._losses[layer] if self._losses.get(layer) else None

    def set_metrics(self, layer: str, entities: list[str]) -> None:
        """
        Set the metrics for a specific output layer.

        Parameters
        ----------
        layer : str
            Name of the output layer.
        entities : list of str
            Names of the metrics.
        """
        self._metrics[layer] = entities

    def get_metrics(self, layer: str) -> t.Metrics:
        """
        Get the metrics for a specific output layer.

        Parameters
        ----------
        layer : str
            Name of the output layer.

        Returns
        -------
        list of str
            Names of the metrics.
        """
        return self._metrics[layer].copy() if self._metrics.get(layer) else list()

    def compile(self) -> None:
        """
        Compile the model.

        Raises
        ------
        ModelError
            If there is an issue compiling the model.
        """
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
        """
        Set the input or output features for a specific layer.

        Parameters
        ----------
        layer : str
            Name of the layer.
        columns : list of str
            Names of the columns.
        at : {'input', 'output'}
            Side to set features for.

        Raises
        ------
        SetError
            If there is an issue setting the features.
        """
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
        """
        Get the input or output features for a specific layer.

        Parameters
        ----------
        layer : str
            Name of the layer.
        at : {'input', 'output'}
            Side to get features for.

        Returns
        -------
        list of str
            Names of the features.
        """
        if at == "input":
            features = self._input_features
        else:
            features = self._output_features

        return features[layer].copy() if features.get(layer) else list()

    def set_callback(self, entity: str, params: t.CallbackParams) -> None:
        """
        Set the callback for the model.

        Parameters
        ----------
        entity : str
            Name of the callback type.
        params : CallbackParams
            Parameters for the callback.

        Raises
        ------
        SetError
            If there is an issue setting the callback.
        """
        try:
            prototype = enums.callbacks.classes[entity]
            self._callbacks[entity] = prototype(**params)
        except KeyError:
            raise errors.SetError("There is no prototype for this callback!")
        except (ValueError, AttributeError, TypeError):
            raise errors.SetError("Unable to set the callback!")

    def get_callback(self, entity: str) -> t.Callback:
        """
        Get the callback for the model.

        Parameters
        ----------
        entity : str
            Name of the callback type.

        Returns
        -------
        Callback or None
            Callback instance.
        """
        return self._callbacks.get(entity)

    def delete_callback(self, entity: str) -> None:
        """
        Delete the callback from the model.

        Parameters
        ----------
        entity : str
            Name of the callback type.
        """
        self._callbacks.pop(entity, None)

    def fit(
        self, data: t.DataFrame, batch_size: int, num_epochs: int, val_split: float
    ) -> None:
        """
        Fit the model to the provided data.

        Parameters
        ----------
        data : DataFrame
            Input and output data.
        batch_size : int
            Batch size.
        num_epochs : int
            Number of epochs.
        val_split : float
            Validation split.

        Raises
        ------
        ModelError
            If there is an issue fitting the model.
        """
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
        """
        Update the training history with new logs.

        Parameters
        ----------
        logs : DataFrame
            New logs to be added to the history.
        """
        history_len = len(self._history)
        logs.insert(0, "epoch", range(history_len + 1, history_len + len(logs) + 1))

        self._history = pd.concat([self._history, logs])

    def plot_history(self, y: t.LogsNames, points: bool) -> t.Chart:
        """
        Plot the training history.

        Parameters
        ----------
        y : list of str
            Names of the logs to plot.
        points : bool
            Whether to include points on the plot.

        Returns
        -------
        Chart
            Altair chart representing the training history.
        """
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
        """Name of the model."""
        return self._name

    @property
    def inputs(self) -> t.Layers:
        """Names of the input layers."""
        return self._inputs.copy()

    @property
    def outputs(self) -> t.Layers:
        """Names of the output layers."""
        return self._outputs.copy()

    @property
    def input_shape(self) -> t.LayerShape:
        """Shapes of the input layers."""
        return self._input_shape.copy()

    @property
    def output_shape(self) -> t.LayerShape:
        """Shapes of the output layers."""
        return self._output_shape.copy()

    @property
    def input_configured(self) -> bool:
        """True if all input layers are configured, False otherwise."""
        return (
            True
            if self._input_configured and all(self._input_configured.values())
            else False
        )

    @property
    def output_configured(self) -> bool:
        """True if all output layers are configured, False otherwise."""
        return (
            True
            if self._output_configured and all(self._output_configured.values())
            else False
        )

    @property
    def _optimizer_is_set(self) -> bool:
        """True if an optimizer is set, False otherwise."""
        return True if self._optimizer else False

    @property
    def _losses_are_set(self) -> bool:
        """True if losses are set for all output layers, False otherwise."""
        return True if self._losses and all(self._losses.values()) else False

    @property
    def built(self) -> bool:
        """True if the model is built, False otherwise."""
        return self._built

    @property
    def compiled(self) -> bool:
        """True if the model is compiled, False otherwise."""
        return self._compiled

    @property
    def history(self) -> t.DataFrame:
        """Training history DataFrame."""
        return self._history.copy()

    @property
    def summary(self) -> None:
        """Summary of the model."""
        self._object.summary()

    @property
    def graph(self) -> bytes:
        """Bytes representation of the model graph."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tf.keras.utils.plot_model(
                self._object, to_file=tmp.name, show_shapes=True, rankdir="LR", dpi=200
            )

            graph = tmp.read()

        return graph

    @property
    def as_bytes(self) -> bytes:
        """Bytes representation of the saved model."""
        with tempfile.NamedTemporaryFile() as tmp:
            self._object.save(filepath=tmp.name, save_format="h5")

            model_as_bytes = tmp.read()

        return model_as_bytes


class UploadedModel(Model):
    """Class representing the uploaded model."""

    def __init__(self) -> None:
        """Initialize an empty uploaded machine learning model."""
        super().__init__()

    def upload(self, buff: io.BytesIO) -> None:
        """
        Upload a model from the provided file.

        Parameters
        ----------
        buff : file-like object
            Byte buffer containing the model.

        Raises
        ------
        UploadError
            If there is an issue reading the model from the file. If there is an issue
            validating the shapes of the model.
        """
        try:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(buff.getbuffer())

                model = typing.cast(
                    t.Object, tf.keras.models.load_model(tmp.name, compile=False)
                )
                tools.model.validate_shapes(model.input_shape)
                tools.model.validate_shapes(model.output_shape)

                self._object = model
                self._built = True
                self._set_config()
                self.update_state()
        except (ValueError, errors.ValidateModelError) as error:
            raise errors.UploadError(error)

    def evaluate(self, data: t.DataFrame, batch_size: int) -> t.EvaluationResults:
        """
        Evaluate the model on the provided data.

        Parameters
        ----------
        data : DataFrame
            Input and output data.
        batch_size : int
            Batch size.

        Raises
        ------
        ModelError
            If there is an issue evaluating the model.

        Returns
        -------
        DataFrame
            Evaluation results as a DataFrame.
        """
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
        """
        Make predictions using the model on the provided data.

        Parameters
        ----------
        data : DataFrame
            Input data.
        batch_size : int
            Batch size.

        Raises
        ------
        ModelError
            If there is an issue making predictions.

        Returns
        -------
        list of DataFrame
            Predictions as DataFrames.
        """
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
    """Class representing the created model."""

    def __init__(self) -> None:
        """Initialize an empty created machine learning model."""
        self.reset_state()

    def reset_state(self) -> None:
        """Reset the state of the created model."""
        super().reset_state()

        self._layers: t.LayerObject = dict()

    def set_name(self, name: str) -> None:
        """
        Set the name of the created model.

        Parameters
        ----------
        name : str
            Name of the model.
        """
        self._name = name

    def set_layer(
        self,
        entity: str,
        name: str,
        params: t.LayerParams,
        connection: t.LayerConnection,
    ) -> None:
        """
        Set the layer for the created model.

        Parameters
        ----------
        entity : str
            Name of the layer type.
        name : str
            Name of the layer.
        params : LayerParams
            Parameters for the layer.
        connection : Layer, list of Layer or None
            Layer(s) to connect.

        Raises
        ------
        SetError
            If there is an issue setting the layer.
        """
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
        """
        Delete the last layer from the created model.

        Raises
        ------
        DeleteError
            If there are no layers to remove.
        """
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
        """
        Set the output layers for the created model.

        Parameters
        ----------
        outputs : list of str
            Names of the layers.

        Raises
        ------
        SetError
            If there is an issue setting the output layers.
        """
        if not outputs:
            raise errors.SetError("Please, select at least one output!")

        self._outputs = outputs

    def create(self) -> None:
        """
        Create the machine learning model.

        Raises
        ------
        CreateError
            If there is an issue creating the model.
        """
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
        """Objects of the layers."""
        return self._layers.copy()
