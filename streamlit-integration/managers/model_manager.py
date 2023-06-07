import io
import os
import tempfile
from typing import Type

import data_classes.model as model_cls
import streamlit as st
import tensorflow as tf
import widgets.layers as wl

import managers.errors as err


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
    """Upload TensorFlow model.

    Parameters
    ----------
    buff : File-like object | None
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
    """Refresh attributes of a model container.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    model.name = model.instance.name
    model.input_layers = {
        name: layer
        for name, layer in zip(model.instance.input_names, model.instance.inputs)
    }
    model.output_layers = {
        name: layer
        for name, layer in zip(model.instance.output_names, model.instance.outputs)
    }
    model.layers = {layer.name: layer.output for layer in model.instance.layers}
    model.input_shapes = {layer.name: layer.shape[1] for layer in model.instance.inputs}
    model.output_shapes = {layer_name: 1 for layer_name in model.instance.output_names}
    model.losses = {name: list() for name in model.output_layers}
    model.metrics = {name: list() for name in model.output_layers}


def add_layer(
    layer_instance: Type[tf.keras.layers.Layer],
    layer_params: wl.LayerParams,
    layer_connection: wl.LayerConnection,
    model: model_cls.Model,
) -> None:
    """Add layer to a model.

    Parameters
    ----------
    layer_instance : Layer
        TensorFlow layer class.
    layer_params : dict
        Dictionary containing construction parameters of a layer.
    layer_connection : str, list, int, or None
        A single layer name or a sequence of layers' names. None if no connection is
        provided. 0 if connection was required, but not given.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model isn't instantiated.
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
        layer = {name: layer_instance(**layer_params)}
        model.input_layers.update(layer)
    else:
        if isinstance(layer_connection, str):
            connect_to = model.layers[layer_connection]
        else:
            connect_to = [model.layers[name] for name in layer_connection]

        layer = {name: layer_instance(**layer_params)(connect_to)}

    model.layers.update(layer)


def set_outputs(outputs: list[str], model: model_cls.Model) -> None:
    """Set outputs for a model.

    Parameters
    ----------
    outputs : list
        List of all output layers' names.
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model isn't instantiated.
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
    """Show model summary.

    Parameters
    ----------
    model : Model
        Model container object.

    Raises
    ------
    NoModelError
        When model isn't instantiated.
    NoOutputLayersError
        When there are no output layers in the model.
    """
    if not model_exists(model):
        raise err.NoModelError("Please, create or upload a model!")

    if not model.output_layers:
        raise err.NoOutputLayersError("Please, set the model outputs!")

    model.instance.summary(print_fn=lambda x: st.text(x))


def download_graph(model: model_cls.Model) -> bytes:
    """Save model graph.

    Parameters
    ----------
    model : Model
        Model container object.

    Returns
    -------
    bytes
        Graph object in bytes.

    Raises
    ------
    NoModelError
        When model isn't instantiated.
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
    """Save model object.

    Parameters
    ----------
    model : Model
        Model container object.

    Returns
    -------
    bytes
        Model object in bytes.

    Raises
    ------
    NoModelError
        When model isn't instantiated.
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
