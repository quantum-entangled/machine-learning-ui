import io
import math
import pytest
import itertools
import collections as clns
import pandas as pd
import plotly as ply
import tensorflow as tf

import mlui.managers.data_manager as dm
import mlui.data_classes.data as data_cls
import mlui.data_classes.model as model_cls
import mlui.widgets.layers as wl
import mlui.managers.errors as err
import mlui.managers.model_manager as mm


def test_upload_file(csv_str: str):
    data = data_cls.Data()
    model = model_cls.Model()
    csv_file = bytes(csv_str, "utf-8")
    with io.BytesIO(csv_file) as buff:
        dm.upload_file(buff, data, model)
    with io.BytesIO(csv_file) as buff:
        file = pd.read_csv(buff, header=0, skipinitialspace=True)

    assert data.file.equals(file)
    assert data.columns == list(data.file.columns)
    assert data.columns_counter == clns.Counter(data.columns)
    assert data.available_columns == data.columns.copy()


def test_error_upload_file(
    data: data_cls.Data,
    model: model_cls.Model,
    empty_csv: str,
    csv_with_multiindex: str,
    csv_with_diff_types: str,
    csv_invalid_delimiter: str,
    csv_invalid_indentations: str,
):
    with pytest.raises(err.UploadError):
        with io.BytesIO(bytes(empty_csv, "utf-8")) as buff:
            dm.upload_file(buff, data, model)

    with pytest.raises(err.UploadError):
        with io.BytesIO(bytes(csv_with_multiindex, "utf-8")) as buff:
            dm.upload_file(buff, data, model)

    with pytest.raises(err.UploadError):
        with io.BytesIO(bytes(csv_with_diff_types, "utf-8")) as buff:
            dm.upload_file(buff, data, model)

    with pytest.raises(err.UploadError):
        with io.BytesIO(bytes(csv_invalid_delimiter, "utf-8")) as buff:
            dm.upload_file(buff, data, model)

    with pytest.raises(err.UploadError):
        with io.BytesIO(bytes(csv_invalid_indentations, "utf-8")) as buff:
            dm.upload_file(buff, data, model)


def test_file_exists(csv_str: str):
    data = data_cls.Data()
    model = model_cls.Model()
    csv_file = bytes(csv_str, "utf-8")
    with io.BytesIO(csv_file) as buff:
        dm.upload_file(buff, data, model)
    with io.BytesIO(csv_file) as buff:
        file = pd.read_csv(buff, header=0, skipinitialspace=True)

    if not file.empty:
        assert dm.file_exists(data) is True
    else:
        assert dm.file_exists(data) is False


def test_show_data_stats(csv_str: str):
    data = data_cls.Data()
    model = model_cls.Model()
    csv_file = bytes(csv_str, "utf-8")
    with io.BytesIO(csv_file) as buff:
        dm.upload_file(buff, data, model)

    data_stats = dm.show_data_stats(data)

    if dm.file_exists(data):
        assert isinstance(data_stats, pd.DataFrame)
    else:
        assert data_stats is None


def test_show_data_plot(csv_str: str):
    data = data_cls.Data()
    model = model_cls.Model()
    csv_file = bytes(csv_str, "utf-8")
    with io.BytesIO(csv_file) as buff:
        dm.upload_file(buff, data, model)
    with io.BytesIO(csv_file) as buff:
        file = pd.read_csv(buff, header=0, skipinitialspace=True)

    columns_names = list(file.columns)

    for i in range(len(columns_names)):
        for j in range(i, len(columns_names)):
            plot = dm.show_data_plot(columns_names[i], columns_names[j], data)
            assert isinstance(plot, ply.graph_objs.Figure)


def test_set_and_get_layer_columns(csv_str: str):
    csv_file = bytes(csv_str, "utf-8")
    with io.BytesIO(csv_file) as buff:
        file = pd.read_csv(buff, header=0, skipinitialspace=True)
    columns_names = list(file.columns)
    columns_num = len(columns_names)

    for input_shape in range(1, columns_num - 1):
        output_shape = columns_num - input_shape

        data = data_cls.Data()
        model = model_cls.Model()
        model_name = "test_model"
        mm.create_model(model_name, model)

        with io.BytesIO(csv_file) as buff:
            dm.upload_file(buff, data, model)

        input_params = wl.InputParams(name="input", shape=(input_shape,))
        mm.add_layer(
            layer_cls=tf.keras.Input,
            layer_params=input_params,
            layer_connection=None,
            model=model,
        )

        dense_params = wl.DenseParams(name="dense", units=32, activation="relu")
        mm.add_layer(
            layer_cls=tf.keras.layers.Dense,
            layer_params=dense_params,
            layer_connection="input",
            model=model,
        )

        output_params = wl.DenseParams(
            name="output", units=output_shape, activation="relu"
        )
        mm.add_layer(
            layer_cls=tf.keras.layers.Dense,
            layer_params=output_params,
            layer_connection="dense",
            model=model,
        )

        mm.set_outputs(["output"], model)

        input_columns = dm.get_layer_columns("Input", "input", data)
        output_columns = dm.get_layer_columns("Output", "output", data)

        assert isinstance(data.input_columns["input"], list)
        assert len(input_columns) == 0
        assert isinstance(data.output_columns["output"], list)
        assert len(output_columns) == 0

        dm.set_columns("Input", "input", [columns_names[0]], data, model)
        assert len(data.available_columns) == columns_num - 1
        assert data.available_columns == columns_names[1:]

        num_of_combinations = math.comb(columns_num, input_shape)
        input_columns_comb = list(itertools.combinations(columns_names, input_shape))

        for i in range(num_of_combinations):
            input_columns = list(input_columns_comb[i])
            output_columns = [
                column for column in columns_names if column not in input_columns
            ]
            assert model.output_shapes["output"] == output_shape
            assert len(output_columns) == output_shape

            dm.set_columns("Input", "input", input_columns, data, model)
            dm.set_columns("Output", "output", output_columns, data, model)

            assert dm.get_layer_columns("Input", "input", data) == input_columns
            assert dm.get_layer_columns("Output", "output", data) == output_columns
            assert len(data.available_columns) == 0
            assert dm.layers_are_filled("Input", data, model) is True
            assert dm.layers_are_filled("Output", data, model) is True


def test_errors_set_columns(data: data_cls.Data, model: model_cls.Model):
    with pytest.raises(err.NoColumnsSelectedError):
        dm.set_columns("Input", "input", list(), data, model)

    model.input_shapes = {"input": 1}
    with pytest.raises(err.LayerOverfilledError):
        dm.set_columns("Input", "input", [1, 2], data, model)


def test_layers_are_filled(data: data_cls.Data, model: model_cls.Model):
    model_name = "test_model"
    mm.create_model(model_name, model)

    input_1_params = wl.InputParams(name="input_1", shape=(1,))
    mm.add_layer(
        layer_cls=tf.keras.Input,
        layer_params=input_1_params,
        layer_connection=None,
        model=model,
    )

    input_2_params = wl.InputParams(name="input_2", shape=(2,))
    mm.add_layer(
        layer_cls=tf.keras.Input,
        layer_params=input_2_params,
        layer_connection=None,
        model=model,
    )

    output_1_params = wl.DenseParams(name="output_1", units=1, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=output_1_params,
        layer_connection="input_1",
        model=model,
    )

    output_2_params = wl.DenseParams(name="output_2", units=2, activation="relu")
    mm.add_layer(
        layer_cls=tf.keras.layers.Dense,
        layer_params=output_2_params,
        layer_connection="input_2",
        model=model,
    )

    mm.set_outputs(["output_1", "output_2"], model)

    dm.set_columns("Input", "input_1", [1], data, model)
    assert dm.layers_are_filled("Input", data, model) is False

    dm.set_columns("Input", "input_2", [2], data, model)
    assert dm.layers_are_filled("Input", data, model) is False

    dm.set_columns("Input", "input_2", [2, 3], data, model)
    assert dm.layers_are_filled("Input", data, model) is True

    dm.set_columns("Output", "output_1", [4], data, model)
    assert dm.layers_are_filled("Output", data, model) is False

    dm.set_columns("Output", "output_2", [5, 6], data, model)
    assert dm.layers_are_filled("Output", data, model) is True


def test_split_data(csv_str: str):
    data = data_cls.Data()
    model = model_cls.Model()
    csv_file = bytes(csv_str, "utf-8")
    with io.BytesIO(csv_file) as buff:
        dm.upload_file(buff, data, model)
    with io.BytesIO(csv_file) as buff:
        file = pd.read_csv(buff, header=0, skipinitialspace=True)

    columns_names = list(file.columns)

    model_name = "test_model"
    mm.create_model(model_name, model)

    layers_shapes = dict()

    if not file.empty and len(columns_names) > 1:
        input_1_params = wl.InputParams(name="input_1", shape=(1,))
        mm.add_layer(
            layer_cls=tf.keras.Input,
            layer_params=input_1_params,
            layer_connection=None,
            model=model,
        )
        layers_shapes["input_1"] = 1

        output_1_params = wl.DenseParams(name="output_1", units=1, activation="relu")
        mm.add_layer(
            layer_cls=tf.keras.layers.Dense,
            layer_params=output_1_params,
            layer_connection="input_1",
            model=model,
        )
        layers_shapes["output_1"] = 1

        if len(columns_names) > 3:
            output_2_params = wl.DenseParams(
                name="output_2", units=2, activation="relu"
            )
            mm.add_layer(
                layer_cls=tf.keras.layers.Dense,
                layer_params=output_2_params,
                layer_connection="input_1",
                model=model,
            )
            layers_shapes["output_2"] = 2

            if len(columns_names) > 5:
                input_2_params = wl.InputParams(name="input_2", shape=(2,))
                mm.add_layer(
                    layer_cls=tf.keras.Input,
                    layer_params=input_2_params,
                    layer_connection=None,
                    model=model,
                )
                layers_shapes["input_2"] = 2

            mm.set_outputs(["output_1", "output_2"], model)
        else:
            mm.set_outputs(["output_1"], model)

        available_columns = columns_names
        for layer, shape in layers_shapes.items():
            columns = available_columns[:shape]
            available_columns = available_columns[shape:]
            if layer[:5] == "input":
                dm.set_columns("Input", layer, columns, data, model)
            else:
                dm.set_columns("Output", layer, columns, data, model)
        test_size = 0.5
        dm.split_data(test_size, data, model)
