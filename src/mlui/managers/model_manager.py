import io
import os
import tempfile
from typing import Literal, Type, TypedDict

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.delta_generator as dg
import tensorflow as tf

import mlui.data_classes.data as data_cls
import mlui.data_classes.model as model_cls
import mlui.managers.data_manager as dm
import mlui.managers.errors as err
import mlui.widgets.callbacks as wc
import mlui.widgets.layers as wl
import mlui.widgets.optimizers as wo


class ChartParams(TypedDict):
    """Type annotation for the chart parameters."""

    scheme: str
    X_ticks: int | float
    Y_ticks: int | float
    X_l_lim: int | float
    X_r_lim: int | float
    Y_l_lim: int | float
    Y_r_lim: int | float
    X_title: str | None
    Y_title: str | None
    legend_or: Literal[
        "left",
        "right",
        "top",
        "bottom",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ]
    legend_dir: Literal["vertical", "horizontal"]
    legend_title: str | None
    height: int | float
    points: bool
    Y_zero: bool
    X_grid: bool
    Y_grid: bool
    X_inter: bool
    Y_inter: bool


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
    except ValueError:
        raise err.UploadError("Unable to upload the model!")


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
    model.metrics = {name: dict() for name in model.output_layers}


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


def set_metrics(
    layer: str,
    metrics_to_set: model_cls.Metrics,
    model: model_cls.Model,
) -> None:
    """Set the metric for the model's output layer.

    Parameters
    ----------
    layer : str
        Name of the layer to which the metric will be attached.
    metrics_to_set : Metrics
        Names and instances of TensorFlow metrics classes.
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

    model.metrics[layer] = metrics_to_set


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

    metrics = {
        layer_name: layer_metrics.values()
        for layer_name, layer_metrics in model.metrics.items()
    }

    model.instance.compile(
        optimizer=model.optimizer, loss=model.losses, metrics=metrics
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


def reset_callbacks(model: model_cls.Model) -> None:
    """Reset all existing callbacks of the model.

    Parameters
    ----------
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    model.callbacks.clear()


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
    FileHasWrongValues
        When the file contains NaN and/or object values.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not (data.input_train_data and data.output_train_data):
        raise err.DataNotSplitError("Please, split the data first!")

    if not model.compiled:
        raise err.ModelNotCompiledError("Please, compile the model first!")

    if data.has_nans:
        raise err.FileHasWrongValues("Please, remove the NaN values from the file!")

    if data.has_object_cols:
        raise err.FileHasWrongValues("Please, encode all the object values!")

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
    df = pd.DataFrame(history.history)
    epochs_num = len(model.training_history)

    df.insert(0, "epoch", range(epochs_num + 1, epochs_num + len(df) + 1))

    model.training_history = pd.concat([model.training_history, df])
    model.trained = True


def show_history_plot(
    Y: list[str], chart_params: ChartParams, model: model_cls.Model
) -> alt.Chart:
    """Show the history plot.

    Parameters
    ----------
    Y : list of str
        Y-axis columns names.
    chart_params : ChartParams
        Parameters for the chart layout.
    model : Model
        Model container object.

    Returns
    -------
    Chart
        Altair chart object.

    Raises
    ------
    NoModelError
        When model is not instantiated.
    ModelNotTrainedError
        When the model is not trained.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.trained:
        raise err.ModelNotTrainedError("Please, train the model first!")

    df = model.training_history.loc[:, ["epoch", *Y]]
    melted_df = df.melt("epoch", var_name="log_name", value_name="log_value")
    chart = (
        alt.Chart(melted_df)
        .mark_line(point=chart_params["points"])
        .encode(
            x=alt.X("epoch")
            .axis(tickCount=chart_params["X_ticks"], grid=chart_params["X_grid"])
            .scale(domain=(chart_params["X_l_lim"], chart_params["X_r_lim"]))
            .title(chart_params["X_title"]),
            y=alt.Y("log_value")
            .axis(tickCount=chart_params["Y_ticks"], grid=chart_params["Y_grid"])
            .scale(
                zero=chart_params["Y_zero"],
                domain=(chart_params["Y_l_lim"], chart_params["Y_r_lim"]),
            )
            .title(chart_params["Y_title"]),
            color=alt.Color("log_name")
            .scale(scheme=chart_params["scheme"])
            .legend(
                orient=chart_params["legend_or"],
                direction=chart_params["legend_dir"],
                title=chart_params["legend_title"],
            ),
        )
        .interactive(bind_x=chart_params["X_inter"], bind_y=chart_params["Y_inter"])
        .properties(height=chart_params["height"])
    )

    return chart


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
