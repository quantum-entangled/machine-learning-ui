import os
import tempfile
import zipfile
from typing import Any

import tensorflow as tf
from data_classes.model import Model

from managers.errors import UploadError


def model_exists(model: Model) -> bool:
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


def upload_model(path: Any, model: Model) -> None:
    """Upload TensorFlow model.

    Parameters
    ----------
    path : Any
        Path to a model file.
    model : Model
        Model container object.
    """
    if not path:
        return

    try:
        model.instance = tf.keras.models.load_model(path)
    except ValueError as error:
        raise UploadError(f"Unable to upload the model!") from error


def refresh_model(model: Model) -> None:
    """Refresh attributes of model container.

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
    model.layers = model.input_layers | model.output_layers
    model.input_shapes = {layer.name: layer.shape[1] for layer in model.instance.inputs}
    model.output_shapes = {layer_name: 1 for layer_name in model.instance.output_names}
    model.losses = {name: list() for name in model.output_layers}
    model.metrics = {name: list() for name in model.output_layers}
