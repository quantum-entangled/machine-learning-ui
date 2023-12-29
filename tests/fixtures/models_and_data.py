import pytest
import tensorflow as tf

import mlui.classes.model as model_cls
import mlui.classes.data as data_cls
import mlui.widgets.layers as wl
import mlui.enums as enums
import mlui.types.classes as t


@pytest.fixture
def data() -> data_cls.Data:
    return data_cls.Data()


@pytest.fixture
def model() -> model_cls.Model:
    return model_cls.Model()


@pytest.fixture
def uploaded_model() -> model_cls.UploadedModel:
    return model_cls.UploadedModel()


@pytest.fixture
def created_model() -> model_cls.CreatedModel:
    return model_cls.CreatedModel()


@pytest.fixture
def not_empty_model() -> model_cls.CreatedModel:
    model = model_cls.CreatedModel()
    model.set_name("test_model")
    earlystopping_params = t.EarlyStoppingParams(min_delta=0, patience=0)
    model.set_callback("EarlyStopping", earlystopping_params)
    layer_params = t.InputParams(shape=(4,))
    model.set_layer("Input", "input", layer_params, None)
    return model


@pytest.fixture
def model_with_layers() -> model_cls.Model:
    model = model_cls.CreatedModel()
    model.set_name("test_model_with_layers")
    input_layer_params = t.InputParams(shape=(4,))
    model.set_layer("Input", "input", input_layer_params, None)
    input_layer = model.layers["input"]
    dense_layer_params = t.DenseParams(units=32, activation="relu")
    model.set_layer("Dense", "dense", dense_layer_params, input_layer)
    dense_layer = model.layers["dense"]
    output_layer_params = t.DenseParams(units=6, activation="relu")
    model.set_layer("Dense", "output", output_layer_params, dense_layer)
    return model


@pytest.fixture
def full_model(model_with_layers: model_cls.Model) -> model_cls.Model:
    model_with_layers.set_outputs(["output"])
    model_with_layers.create()
    return model_with_layers


@pytest.fixture
def compiled_model(full_model: model_cls.Model) -> model_cls.Model:
    adam_params = t.AdamParams(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    full_model.set_optimizer("Adam", adam_params)
    full_model.set_loss("output", "MeanAbsoluteError")
    full_model.set_metrics("output", ["MeanAbsoluteError"])
    full_model.compile()
    return full_model
