import pytest
import tensorflow as tf

from mlui.data_classes import data as data_cls
from mlui.data_classes import model as model_cls
from mlui.managers import model_manager as mm
import mlui.widgets.layers as wl


@pytest.fixture
def data() -> data_cls.Data:
    return data_cls.Data()


@pytest.fixture
def model() -> model_cls.Model:
    return model_cls.Model()


@pytest.fixture
def not_empty_model() -> model_cls.Model:
    model = model_cls.Model()
    model_name = "test_model"
    model.name = model_name
    model.instance = tf.keras.Model(inputs=list(), outputs=list(), name=model_name)
    return model


@pytest.fixture
def model_with_layers() -> model_cls.Model:
    model = model_cls.Model()
    model_name = "test_model"
    mm.create_model(model_name, model)

    input_params = wl.InputParams(name="input", shape=(4,))
    mm.add_layer(
        layer_cls=tf.keras.Input,
        layer_params=input_params,
        layer_connection=None,
        model=model,
    )

    dense_1_params = wl.DenseParams(name="dense_1", units=32, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=dense_1_params,
        layer_connection="input",
        model=model,
    )

    dense_2_params = wl.DenseParams(name="dense_2", units=32, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=dense_2_params,
        layer_connection="input",
        model=model,
    )

    concatenate_params = wl.LayerParams(name="concatenate")
    mm.add_layer(
        layer_cls=tf.keras.layers.Concatenate,
        layer_params=concatenate_params,
        layer_connection=["dense_1", "dense_2"],
        model=model,
    )

    output_params = wl.DenseParams(name="output", units=32, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=output_params,
        layer_connection="concatenate",
        model=model,
    )

    return model
