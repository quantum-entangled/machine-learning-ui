import io
import os
import tempfile
from typing import Type

import data_classes.data as data_cls
import data_classes.model as model_cls
import managers.data_manager as dm
import managers.errors as err
import plotly as ply
import plotly.express as px
import streamlit as st
import streamlit.delta_generator as dg
import tensorflow as tf
import widgets.callbacks as wc
import widgets.layers as wl
import widgets.optimizers as wo


def model_exists(model: model_cls.Model) -> bool:
    """Check if model exists.

    Parameters
    ----------
    model : Model
        Model container object.

    Returns
    -------
    bool
        True if model exists, False otherwise.
    """
    return True if model.instance else False


def create_model(model_name: str, model: model_cls.Model) -> None:
    """Create a new model.

    Parameters
    ----------
    model_name : str
        Model's name.
    model : Model
        Model container object.

    Raises
    ------
    NoModelNameError
        When trying to create a model with no name.
    SameModelNameError
        When trying to create a model with already existing name.
    """
    if not model_name:
        raise err.NoModelNameError("Please, enter the model name!")

    if model.name == model_name:
        raise err.SameModelNameError(
            "Model with this name already exists! Please, enter a different name!"
        )

    model.instance = tf.keras.Model(inputs=list(), outputs=list(), name=model_name)
    refresh_model(model)


def upload_model(buff: io.BytesIO | None, model: model_cls.Model) -> None:
    """Upload a TensorFlow model.

    Parameters
    ----------
    buff : File-like object or None
        Buffer object to upload.
    model : Model
        Model container object.

    Raises
    ------
    UploadError
        For errors during the uploading procedure.
    """
    if not buff:
        return

    try:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(buff.getbuffer())
            model.instance = tf.keras.models.load_model(tmp.name)
            refresh_model(model)
    except ValueError as error:
        raise err.UploadError("Unable to upload the model!") from error


def refresh_model(model: model_cls.Model) -> None:
    """Refresh the attributes of the model container.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    input_names = model.instance.input_names
    output_names = model.instance.output_names
    inputs = model.instance.inputs
    outputs = model.instance.outputs

    model.name = model.instance.name
    model.input_layers = {name: layer for name, layer in zip(input_names, inputs)}
    model.output_layers = {name: layer for name, layer in zip(output_names, outputs)}
    model.layers = {layer.name: layer.output for layer in model.instance.layers}
    model.input_shapes = {
        name: layer.shape[1] for name, layer in zip(input_names, inputs)
    }
    model.output_shapes = {
        name: layer.shape[1] for name, layer in zip(output_names, outputs)
    }
    model.losses = {name: list() for name in model.output_layers}
    model.metrics = {name: list() for name in model.output_layers}


def add_layer(
    layer_cls: Type[tf.keras.layers.Layer],
    layer_params: wl.LayerParams,
    layer_connection: wl.LayerConnection,
    model: model_cls.Model,
) -> None:
    """Add the layer to the model.

    Parameters
    ----------
    layer_cls : Layer
        TensorFlow layer class.
    layer_params : dict
        Dictionary containing construction parameters of the layer.
    layer_connection : str, list of str, int, or None
        A single layer name or a sequence of layers' names. None if no connection is
        provided. 0 if connection was required, but not given.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    NoLayerNameError
        When trying to create a layer with no name.
    SameLayerNameError
        When trying to create a layer with already existing name.
    NoConnectionError
        When no connection is provided for the layer.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    name = layer_params["name"]

    if not name:
        raise err.NoLayerNameError("Please, enter the layer name!")

    if name in model.layers:
        raise err.SameLayerNameError(
            "Layer with this name already exists. Please, enter a different name!"
        )

    if isinstance(layer_connection, int):
        raise err.NoConnectionError("Please, select the connection!")

    if layer_connection is None:
        layer = {name: layer_cls(**layer_params)}
        model.input_layers.update(layer)
    else:
        if isinstance(layer_connection, str):
            connect_to = model.layers[layer_connection]
        else:
            connect_to = [model.layers[name] for name in layer_connection]

        layer = {name: layer_cls(**layer_params)(connect_to)}

    model.layers.update(layer)


def set_outputs(outputs: list[str], model: model_cls.Model) -> None:
    """Set the outputs for the model.

    Parameters
    ----------
    outputs : list of str
        List of all output layers' names.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    NoOutputsSelectedError
        When no layers are selected for the model outputs.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not outputs:
        raise err.NoOutputsSelectedError("Please, select the model outputs!")

    model.output_layers = {name: model.layers[name] for name in outputs}
    model.instance = tf.keras.Model(
        inputs=model.input_layers, outputs=model.output_layers, name=model.name
    )
    refresh_model(model)


def show_summary(model: model_cls.Model) -> None:
    """Show the model summary.

    Parameters
    ----------
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    NoOutputLayersError
        When there are no output layers in the model.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.output_layers:
        raise err.NoOutputLayersError("Please, set the model outputs!")

    model.instance.summary(print_fn=lambda x: st.text(x))


def download_graph(model: model_cls.Model) -> bytes:
    """Save the model graph.

    Parameters
    ----------
    model : Model
        Model container object.

    Returns
    -------
    bytes
        Graph representation in PDF format as bytes object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    NoOutputLayersError
        When there are no output layers in the model.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.output_layers:
        raise err.NoOutputLayersError("Please, set the model outputs!")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tf.keras.utils.plot_model(
            model.instance,
            to_file=tmp.name,
            show_shapes=True,
            rankdir="LR",
            dpi=200,
        )
        graph = tmp.read()
        tmp.close()
        os.unlink(tmp.name)

    return graph


def download_model(model: model_cls.Model) -> bytes:
    """Save the model object.

    Parameters
    ----------
    model : Model
        Model container object.

    Returns
    -------
    bytes
        Model representation in HDF5 format as bytes object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    NoOutputLayersError
        When there are no output layers in the model.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.output_layers:
        raise err.NoOutputLayersError("Please, set the model outputs!")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        model.instance.save(
            filepath=tmp.name,
            save_format="h5",
        )
        model_object = tmp.read()
        tmp.close()
        os.unlink(tmp.name)

    return model_object


def set_optimizer(
    optimizer_cls: tf.keras.optimizers.Optimizer,
    optimizer_params: wo.OptimizerParams,
    model: model_cls.Model,
) -> None:
    """Set the optimizer for the model.

    Parameters
    ----------
    optimizer_cls : Optimizer
        TensorFlow optimizer class.
    optimizer_params : dict
        Dictionary containing construction parameters of the optimizer.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    NoOutputLayersError
        When there are no output layers in the model.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.output_layers:
        raise err.NoOutputLayersError("Please, set the model outputs!")

    model.optimizer = optimizer_cls(**optimizer_params)


def set_loss(
    layer: str,
    loss_cls: Type[tf.keras.losses.Loss],
    model: model_cls.Model,
) -> None:
    """Set the loss function for the model's output layer.

    Parameters
    ----------
    layer : str
        Name of the layer to which the loss function will be attached.
    loss_cls : Loss
        TensorFlow loss function class.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    NoOutputLayersError
        When there are no output layers in the model.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.output_layers:
        raise err.NoOutputLayersError("Please, set the model outputs!")

    model.losses[layer] = loss_cls()


def set_metric(
    layer: str,
    metric_cls: Type[tf.keras.metrics.Metric],
    model: model_cls.Model,
) -> None:
    """Set the metric for the model's output layer.

    Parameters
    ----------
    layer : str
        Name of the layer to which the metric will be attached.
    metric_cls : Metric
        TensorFlow metric class.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    NoOutputLayersError
        When there are no output layers in the model.
    SameMetricError
        When trying to set the already attached metric to the layer.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.output_layers:
        raise err.NoOutputLayersError("Please, set the model outputs!")

    if metric_cls in [type(metric) for metric in model.metrics[layer]]:
        raise err.SameMetricError("Please, select the distinct metric!")

    model.metrics[layer].append(metric_cls())


def compile_model(model: model_cls.Model) -> None:
    """Compile the TensorFlow model.

    Parameters
    ----------
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    NoOptimizerError
        When no optimizer is set for the model.
    NoLossError
        When loss functions are not set for each layer.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.optimizer:
        raise err.NoOptimizerError("Please, set the model optimizer!")

    if not all(model.losses.values()):
        raise err.NoLossError("Please, set the loss function for each output layer!")

    model.instance.compile(
        optimizer=model.optimizer, loss=model.losses, metrics=model.metrics
    )
    model.compiled = True


def set_callback(
    callback_cls: Type[tf.keras.callbacks.Callback],
    callback_params: wc.CallbackParams,
    model: model_cls.Model,
) -> None:
    """Set callbacks for the model.

    Parameters
    ----------
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    SameCallbackError
        When trying to set the already attached callback to the model.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if callback_cls in [type(callback) for callback in model.callbacks]:
        raise err.SameCallbackError("Please, select the distinct callback!")

    model.callbacks.append(callback_cls(**callback_params))


def fit_model(
    batch_size: int,
    num_epochs: int,
    val_split: float,
    batch_container: dg.DeltaGenerator,
    epoch_container: dg.DeltaGenerator,
    data: data_cls.Data,
    model: model_cls.Model,
) -> None:
    """Fit the TensorFlow model.

    Parameters
    ----------
    batch_size : int
        Batch size hyperparameter for the fitting process.
    num_epochs : int
        Number of epochs hyperparameter for the fitting process.
    val_split : float
        Validation split hyperparameter for the fitting process.
    batch_container : Container
        Streamlit container for displaying the batch logs.
    epoch_container : Container
        Streamlit container for displaying the epoch logs.
    data : Data
        Data container object.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    DataNotSplitError
        When the dataset is not split into training and testing sets.
    ModelNotCompiledError
        When the model is not compiled.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not (data.input_train_data and data.output_train_data):
        raise err.DataNotSplitError("Please, split the data first!")

    if not model.compiled:
        raise err.ModelNotCompiledError("Please, compile the model first!")

    batch_callback = tf.keras.callbacks.LambdaCallback(
        on_batch_end=lambda batch, logs: batch_container.write(
            {"batch": batch + 1} | logs
        )
    )
    epoch_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: epoch_container.write(
            {"epoch": epoch + 1} | logs
        )
    )

    history = model.instance.fit(
        x=data.input_train_data,
        y=data.output_train_data,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=val_split,
        callbacks=model.callbacks + [batch_callback, epoch_callback],
    )
    model.training_history = history.history


def show_history_plot(
    y: str, color: str, model: model_cls.Model
) -> ply.graph_objs.Figure:
    """Show the history plot.

    Parameters
    ----------
    y : str
        Y-axis column name.
    color : str
        Color of the line.
    model : Model
        Model container object.

    Returns
    -------
    Figure
        Plotly figure object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    ModelNotTrainedError
        When the model is not trained.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.training_history:
        raise err.ModelNotTrainedError("Please, train the model first!")

    y_data = model.training_history[y]
    x_data = range(1, len(y_data) + 1)
    fig = px.line(x=x_data, y=y_data, labels={"x": "Epoch", "y": y}, markers=True)
    fig.update_traces(line_color=color)

    return fig


def evaluate_model(
    batch_size: int,
    data: data_cls.Data,
    model: model_cls.Model,
) -> dict[str, float]:
    """Evaluate the TensorFlow model.

    Parameters
    ----------
    batch_size : int
        Batch size hyperparameter for the evaluation process.
    data : Data
        Data container object.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    DataNotSplitError
        When the dataset is not split into training and testing sets.
    ModelNotCompiledError
        When the model is not compiled.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not (data.input_train_data and data.output_train_data):
        raise err.DataNotSplitError("Please, split the data first!")

    if not model.compiled:
        raise err.ModelNotCompiledError("Please, compile the model first!")

    results = model.instance.evaluate(
        x=data.input_test_data,
        y=data.output_test_data,
        batch_size=batch_size,
        callbacks=model.callbacks,
        return_dict=True,
        verbose=0,
    )

    return results


def make_predictions(
    batch_size: int,
    data: data_cls.Data,
    model: model_cls.Model,
) -> dict[str, list[float]]:
    """Make predictions of the TensorFlow model.

    Parameters
    ----------
    batch_size : int
        Batch size hyperparameter for the evaluation process.
    data : Data
        Data container object.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    InputsUnderfilledError
        When some of the input layers are not filled with the data columns.
    ModelNotCompiledError
        When the model is not compiled.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not dm.layers_are_filled("Input", data, model):
        raise err.InputsUnderfilledError(
            "Please, set the data columns for all the input layers!"
        )

    if not model.compiled:
        raise err.ModelNotCompiledError("Please, compile the model first!")

    predictions = model.instance.predict(
        x={name: data.file[values] for name, values in data.input_columns.items()},
        batch_size=batch_size,
        callbacks=model.callbacks,
        verbose=0,
    )
    predictions = {layer: list(value.flatten()) for layer, value in predictions.items()}

    return predictions
