import io
import pytest
import tensorflow as tf
import pandas as pd
import streamlit as st
import plotly as ply

from mlui.data_classes import model as model_cls
from mlui.data_classes import data as data_cls
from mlui.managers import model_manager as mm
from mlui.managers import data_manager as dm

import mlui.managers.errors as err
import mlui.widgets.callbacks as wc
import mlui.widgets.layers as wl
import mlui.widgets.optimizers as wo
import mlui.enums.metrics as em
import mlui.enums.losses as el


def test_model_exists(model: model_cls.Model) -> None:
    exists = mm.model_exists(model)

    assert exists is False


def test_create_model(model: model_cls.Model):
    model_name = "test_model"
    mm.create_model(model_name, model)
    exists = mm.model_exists(model)

    assert exists is True
    assert type(model.instance.inputs) == list
    assert type(model.instance.outputs) == list
    assert model.instance.name == model_name

    with pytest.raises(err.NoModelNameError):
        mm.create_model("", model)

    with pytest.raises(err.SameModelNameError):
        mm.create_model(model_name, model)


def test_download_and_upload_model(
    model: model_cls.Model, model_with_layers: model_cls.Model
):
    mm.create_model("test model 1", model)

    mm.set_outputs(["output"], model_with_layers)
    model_object = mm.download_model(model_with_layers)

    with io.BytesIO(model_object) as model_object_bytes:
        mm.upload_model(model_object_bytes, model)
        assert len(model.input_layers) == len(model_with_layers.input_layers)
        assert len(model.output_layers) == len(model_with_layers.output_layers)
        assert len(model.layers) == len(model_with_layers.layers)


def test_refresh_model(model: model_cls.Model):
    model_name = "test_model"
    model.instance = tf.keras.Model(inputs=list(), outputs=list(), name=model_name)
    mm.refresh_model(model)

    assert model.name == model_name
    assert model.input_layers == dict()
    assert model.output_layers == dict()
    assert model.layers == dict()
    assert model.input_shapes == dict()
    assert model.output_shapes == dict()
    assert model.losses == dict()
    assert model.metrics == dict()


def test_add_layer(not_empty_model: model_cls.Model):
    input_params = wl.InputParams(name="input", shape=(4,))
    mm.add_layer(
        layer_cls=tf.keras.Input,
        layer_params=input_params,
        layer_connection=None,
        model=not_empty_model,
    )

    dense_1_params = wl.DenseParams(name="dense_1", units=32, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=dense_1_params,
        layer_connection="input",
        model=not_empty_model,
    )

    dense_2_params = wl.DenseParams(name="dense_2", units=32, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=dense_2_params,
        layer_connection="input",
        model=not_empty_model,
    )

    concatenate_params = wl.LayerParams(name="concatenate")
    mm.add_layer(
        layer_cls=tf.keras.layers.Concatenate,
        layer_params=concatenate_params,
        layer_connection=["dense_1", "dense_2"],
        model=not_empty_model,
    )

    output_params = wl.DenseParams(name="output", units=32, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=output_params,
        layer_connection="concatenate",
        model=not_empty_model,
    )

    assert len(not_empty_model.layers) == 5

    with pytest.raises(err.NoLayerNameError):
        dense_3_params = wl.DenseParams(name="", units=32, activation="relu")
        mm.add_layer(
            layer_cls=tf.keras.layers.Dense,
            layer_params=dense_3_params,
            layer_connection="input",
            model=not_empty_model,
        )

    with pytest.raises(err.SameLayerNameError):
        dense_2_params = wl.DenseParams(name="dense_2", units=32, activation="relu")
        mm.add_layer(
            layer_cls=tf.keras.layers.Dense,
            layer_params=dense_2_params,
            layer_connection="input",
            model=not_empty_model,
        )

    with pytest.raises(err.NoConnectionError):
        dense_3_params = wl.DenseParams(name="dense_3", units=32, activation="relu")
        mm.add_layer(
            layer_cls=tf.keras.layers.Dense,
            layer_params=dense_3_params,
            layer_connection=0,
            model=not_empty_model,
        )


def test_set_outputs(model: model_cls.Model, model_with_layers: model_cls.Model):
    with pytest.raises(err.NoModelError):
        mm.set_outputs([], model)

    with pytest.raises(err.NoOutputsSelectedError):
        mm.set_outputs([], model_with_layers)

    mm.set_outputs(["output"], model_with_layers)
    assert len(model_with_layers.output_layers) == 1

    output_params = wl.DenseParams(name="output_1", units=32, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=output_params,
        layer_connection="concatenate",
        model=model_with_layers,
    )

    mm.set_outputs(["output", "output_1"], model_with_layers)
    assert len(model_with_layers.output_layers) == 2

    assert len(model_with_layers.instance.inputs) == len(model_with_layers.input_layers)
    for input_name in model_with_layers.input_layers:
        assert (
            model_with_layers.input_layers[input_name]
            in model_with_layers.instance.inputs
        )
        assert any(
            model_with_layers.input_layers[input_name] is input
            for input in model_with_layers.instance.inputs
        )

    assert len(model_with_layers.instance.outputs) == len(
        model_with_layers.output_layers
    )
    for output_name in model_with_layers.output_layers:
        assert any(
            model_with_layers.output_layers[output_name] is output
            for output in model_with_layers.instance.outputs
        )

    assert model_with_layers.instance.name == model_with_layers.name


def test_show_summary(model: model_cls.Model, model_with_layers: model_cls.Model):
    with pytest.raises(err.NoModelError):
        mm.show_summary(model)

    with pytest.raises(err.NoOutputLayersError):
        mm.show_summary(model_with_layers)

    mm.set_outputs(["output"], model_with_layers)
    mm.show_summary(model_with_layers)


def test_download_graph(model: model_cls.Model, model_with_layers: model_cls.Model):
    with pytest.raises(err.NoModelError):
        mm.download_graph(model)

    with pytest.raises(err.NoOutputLayersError):
        mm.download_graph(model_with_layers)

    mm.set_outputs(["output"], model_with_layers)
    graph = mm.download_graph(model_with_layers)

    assert graph is not None


def test_set_optimizer(model: model_cls.Model, model_with_layers: model_cls.Model):
    optimizer_cls = tf.keras.optimizers.Optimizer("optimizer name")
    optimizer_params = wo.OptimizerParams()
    with pytest.raises(err.NoModelError):
        mm.set_optimizer(optimizer_cls, optimizer_params, model)

    with pytest.raises(err.NoOutputLayersError):
        mm.set_optimizer(optimizer_cls, optimizer_params, model_with_layers)

    mm.set_outputs(["output"], model_with_layers)

    adam_params = wo.AdamParams(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    mm.set_optimizer(
        optimizer_cls=tf.keras.optimizers.Adam,
        optimizer_params=adam_params,
        model=model_with_layers,
    )

    assert isinstance(model_with_layers.optimizer, tf.keras.optimizers.Adam)

    rmsprop_params = wo.RMSpropParams(learning_rate=0.001, rho=0.9, momentum=0.0)
    mm.set_optimizer(
        optimizer_cls=tf.keras.optimizers.RMSprop,
        optimizer_params=rmsprop_params,
        model=model_with_layers,
    )

    assert isinstance(model_with_layers.optimizer, tf.keras.optimizers.RMSprop)

    sgd_params = wo.SGDParams(learning_rate=0.01, momentum=0.0)
    mm.set_optimizer(
        optimizer_cls=tf.keras.optimizers.SGD,
        optimizer_params=sgd_params,
        model=model_with_layers,
    )

    assert isinstance(model_with_layers.optimizer, tf.keras.optimizers.SGD)


def test_set_loss(model: model_cls.Model, model_with_layers: model_cls.Model):
    with pytest.raises(err.NoModelError):
        mm.set_loss(layer="output", loss_cls=tf.keras.losses.Loss, model=model)

    with pytest.raises(err.NoOutputLayersError):
        mm.set_loss(
            layer="output", loss_cls=tf.keras.losses.Loss, model=model_with_layers
        )

    mm.set_outputs(["output"], model_with_layers)

    losses = list(el.classes.values())

    for loss in losses:
        mm.set_loss(layer="output", loss_cls=loss, model=model_with_layers)

        assert isinstance(model_with_layers.losses["output"], loss)


def test_set_metric(model: model_cls.Model, model_with_layers: model_cls.Model):
    with pytest.raises(err.NoModelError):
        mm.set_metric(layer="output", metric_cls=tf.keras.metrics.Metric, model=model)

    with pytest.raises(err.NoOutputLayersError):
        mm.set_metric(
            layer="output", metric_cls=tf.keras.metrics.Metric, model=model_with_layers
        )

    mm.set_outputs(["output"], model_with_layers)

    metrics = list(em.classes.values())

    for indx in range(len(metrics)):
        mm.set_metric(layer="output", metric_cls=metrics[indx], model=model_with_layers)

        assert isinstance(model_with_layers.metrics["output"][indx], metrics[indx])

    assert len(model_with_layers.metrics["output"]) == len(metrics)


def test_compile_model(model: model_cls.Model, model_with_layers: model_cls.Model):
    with pytest.raises(err.NoModelError):
        mm.compile_model(model)

    mm.set_outputs(["output"], model_with_layers)

    with pytest.raises(err.NoOptimizerError):
        mm.compile_model(model_with_layers)

    adam_params = wo.AdamParams(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    mm.set_optimizer(
        optimizer_cls=tf.keras.optimizers.Adam,
        optimizer_params=adam_params,
        model=model_with_layers,
    )

    with pytest.raises(err.NoLossError):
        mm.compile_model(model_with_layers)

    mm.set_loss(
        layer="output",
        loss_cls=tf.keras.losses.MeanAbsoluteError,
        model=model_with_layers,
    )

    mm.set_metric(
        layer="output",
        metric_cls=tf.keras.metrics.MeanAbsoluteError,
        model=model_with_layers,
    )

    mm.compile_model(model_with_layers)

    assert model_with_layers.compiled is True
    assert isinstance(model_with_layers.instance.optimizer, tf.keras.optimizers.Adam)
    assert isinstance(
        model_with_layers.instance.loss["output"], tf.keras.losses.MeanAbsoluteError
    )
    assert isinstance(
        model_with_layers.instance.compiled_metrics._metrics["output"][0],
        tf.keras.metrics.MeanAbsoluteError,
    )


def test_set_callback(model: model_cls.Model, model_with_layers: model_cls.Model):
    with pytest.raises(err.NoModelError):
        mm.set_callback(
            callback_cls=tf.keras.callbacks.Callback,
            callback_params=wc.CallbackParams,
            model=model,
        )

    earlystopping_params = wc.EarlyStoppingParams(min_delta=0, patience=0)
    mm.set_callback(
        callback_cls=tf.keras.callbacks.EarlyStopping,
        callback_params=earlystopping_params,
        model=model_with_layers,
    )

    mm.set_callback(
        callback_cls=tf.keras.callbacks.TerminateOnNaN,
        callback_params=wc.CallbackParams(),
        model=model_with_layers,
    )

    assert len(model_with_layers.callbacks) == 2
    assert isinstance(model_with_layers.callbacks[0], tf.keras.callbacks.EarlyStopping)
    assert isinstance(model_with_layers.callbacks[1], tf.keras.callbacks.TerminateOnNaN)

    with pytest.raises(err.SameCallbackError):
        mm.set_callback(
            callback_cls=tf.keras.callbacks.TerminateOnNaN,
            callback_params=wc.CallbackParams(),
            model=model_with_layers,
        )


def test_training(data: data_cls.Data, model: model_cls.Model, csv_str: str):
    csv_file = bytes(csv_str, "utf-8")
    with io.BytesIO(csv_file) as buff:
        dm.upload_file(buff, data, model)
    with io.BytesIO(csv_file) as buff:
        file = pd.read_csv(buff, header=0, skipinitialspace=True)

    columns_names = list(file.columns)

    model_name = "test_model"
    mm.create_model(model_name, model)

    if not file.empty:
        input_params = wl.InputParams(name="input", shape=(1,))
        mm.add_layer(
            layer_cls=tf.keras.Input,
            layer_params=input_params,
            layer_connection=None,
            model=model,
        )

        output_params = wl.DenseParams(name="output", units=1, activation="relu")
        mm.add_layer(
            layer_cls=tf.keras.layers.Dense,
            layer_params=output_params,
            layer_connection="input",
            model=model,
        )

        mm.set_outputs(["output"], model)

        adam_params = wo.AdamParams(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        mm.set_optimizer(
            optimizer_cls=tf.keras.optimizers.Adam,
            optimizer_params=adam_params,
            model=model,
        )

        mm.set_loss(
            layer="output", loss_cls=tf.keras.losses.MeanAbsoluteError, model=model
        )

        mm.set_metric(
            layer="output", metric_cls=tf.keras.metrics.MeanAbsoluteError, model=model
        )

        mm.compile_model(model)

        mm.set_callback(
            callback_cls=tf.keras.callbacks.TerminateOnNaN,
            callback_params=wc.CallbackParams(),
            model=model,
        )

        dm.set_columns("Input", "input", [columns_names[0]], data, model)
        dm.set_columns("Output", "output", [columns_names[1]], data, model)

        test_size = 0.5
        dm.split_data(test_size, data, model)

        batch_size = 4
        num_epochs = 5
        val_split = 0.1
        batch_container = st.empty()
        epoch_container = st.expander("Epochs Logs")
        mm.fit_model(
            batch_size,
            num_epochs,
            val_split,
            batch_container,
            epoch_container,
            data,
            model,
        )

        assert isinstance(
            mm.show_history_plot(list(model.training_history)[0], "#0000FF", model),
            ply.graph_objs.Figure,
        )

        results = mm.evaluate_model(batch_size, data, model)
        assert isinstance(results, dict)

        predictions = mm.make_predictions(batch_size, data, model)
        assert isinstance(predictions, dict)


def test_training_errors(model: model_cls.Model, data: data_cls.Data):
    batch_size = 4
    num_epochs = 5
    val_split = 0.1
    batch_container = st.empty()
    epoch_container = st.expander("Epochs Logs")

    with pytest.raises(err.NoModelError):
        mm.fit_model(
            batch_size,
            num_epochs,
            val_split,
            batch_container,
            epoch_container,
            data,
            model,
        )

    with pytest.raises(err.NoModelError):
        mm.show_history_plot([0], "b", model)

    with pytest.raises(err.NoModelError):
        mm.evaluate_model(batch_size, data, model)

    with pytest.raises(err.NoModelError):
        mm.make_predictions(batch_size, data, model)

    model_name = "test_model"
    mm.create_model(model_name, model)

    with pytest.raises(err.ModelNotTrainedError):
        mm.show_history_plot([0], "b", model)

    with pytest.raises(err.DataNotSplitError):
        mm.fit_model(
            batch_size,
            num_epochs,
            val_split,
            batch_container,
            epoch_container,
            data,
            model,
        )

    with pytest.raises(err.DataNotSplitError):
        mm.evaluate_model(batch_size, data, model)

    input_params = wl.InputParams(name="input", shape=(1,))
    mm.add_layer(
        layer_cls=tf.keras.Input,
        layer_params=input_params,
        layer_connection=None,
        model=model,
    )

    output_params = wl.DenseParams(name="output", units=1, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=output_params,
        layer_connection="input",
        model=model,
    )

    mm.set_outputs(["output"], model)

    csv_str = "0,1\n0,0\n0,0"
    csv_file = bytes(csv_str, "utf-8")
    with io.BytesIO(csv_file) as buff:
        dm.upload_file(buff, data, model)
    with io.BytesIO(csv_file) as buff:
        file = pd.read_csv(buff, header=0, skipinitialspace=True)

    columns_names = list(file.columns)

    with pytest.raises(err.InputsUnderfilledError):
        mm.make_predictions(batch_size, data, model)

    dm.set_columns("Input", "input", columns_names[0], data, model)
    dm.set_columns("Output", "output", columns_names[1], data, model)

    test_size = 0.5
    dm.split_data(test_size, data, model)

    with pytest.raises(err.ModelNotCompiledError):
        mm.fit_model(
            batch_size,
            num_epochs,
            val_split,
            batch_container,
            epoch_container,
            data,
            model,
        )

    with pytest.raises(err.ModelNotCompiledError):
        mm.evaluate_model(batch_size, data, model)

    with pytest.raises(err.ModelNotCompiledError):
        mm.make_predictions(batch_size, data, model)
