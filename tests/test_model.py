import io
import pytest
import pandas as pd
import altair as alt
import tensorflow as tf
import mlui.classes.model as model_cls
import mlui.classes.errors as err
import mlui.enums as enums
import mlui.types.classes as t


class TestUpload:
    def test_upload_model(
        self,
        uploaded_model: model_cls.UploadedModel,
        full_model: model_cls.CreatedModel,
    ):
        h5_file = full_model.as_bytes
        with io.BytesIO(h5_file) as buff:
            uploaded_model.upload(buff)

        assert uploaded_model.name == full_model.name
        assert uploaded_model.inputs == full_model.inputs
        assert uploaded_model.input_shape == full_model.input_shape
        assert uploaded_model.outputs == full_model.outputs
        assert uploaded_model.output_shape == full_model.output_shape
        assert uploaded_model.built is True


class TestCreate:
    def test_set_name(self, created_model: model_cls.CreatedModel):
        name = "test_model"
        created_model.set_name(name)
        assert created_model.name is name

    def test_set_layers(self, created_model: model_cls.CreatedModel):
        input_layer_params = t.InputParams(shape=(4,))
        created_model.set_layer("Input", "input", input_layer_params, None)
        assert len(created_model.layers) == 1
        assert "input" in created_model.layers.keys()

        input_layer = created_model.layers["input"]
        dense_layer_params = t.DenseParams(units=32, activation="relu")
        created_model.set_layer("Dense", "dense", dense_layer_params, input_layer)
        assert len(created_model.layers) == 2
        assert "dense" in created_model.layers.keys()

        with pytest.raises(err.SetError):
            dense_layer = created_model.layers["dense"]
            created_model.set_layer("Dense", "dense", dense_layer_params, dense_layer)

        with pytest.raises(err.SetError):
            dense_layer = created_model.layers["dense"]
            dense_1_layer_params = t.DenseParams()
            created_model.set_layer(
                "Dense", "dense_1", dense_1_layer_params, dense_layer
            )

        dense_layer = created_model.layers["dense"]
        created_model.set_layer("Dense", "dense_1", dense_layer_params, dense_layer)

        dense_layer = created_model.layers["dense"]
        dense_1_layer = created_model.layers["dense_1"]
        concatenate_layer_params = t.LayerParams()
        created_model.set_layer(
            "Concatenate",
            "concatenate",
            concatenate_layer_params,
            [dense_layer, dense_1_layer],
        )
        assert "concatenate" in created_model.layers.keys()

        concatenate_layer = created_model.layers["concatenate"]
        BN_layer_params = t.BatchNormalizationParams(momentum=0.99, epsilon=0.001)
        created_model.set_layer(
            "BatchNormalization", "BN", BN_layer_params, concatenate_layer
        )
        assert "BN" in created_model.layers.keys()

        BN_layer = created_model.layers["BN"]
        dropout_params = t.DropoutParams(rate=0.2)
        created_model.set_layer("Dropout", "dropout", dropout_params, BN_layer)
        assert "dropout" in created_model.layers.keys()

    def test_delete_last_layer(self, created_model: model_cls.CreatedModel):
        input_layer_params = t.InputParams(shape=(4,))
        created_model.set_layer("Input", "input", input_layer_params, None)
        input_layer = created_model.layers["input"]
        dense_layer_params = t.DenseParams(units=32, activation="relu")
        created_model.set_layer("Dense", "dense", dense_layer_params, input_layer)
        assert len(created_model.layers) == 2

        created_model.delete_last_layer()
        assert len(created_model.layers) == 1
        assert not "dense" in created_model.layers.keys()
        assert "input" in created_model.layers.keys()
        with pytest.raises(err.DeleteError):
            created_model.delete_last_layer()
            created_model.delete_last_layer()

    def test_set_outputs(self, model_with_layers: model_cls.CreatedModel):
        with pytest.raises(err.SetError):
            model_with_layers.set_outputs([])

        model_with_layers.set_outputs(["output"])
        assert model_with_layers.outputs[0] == "output"

    def test_create_model(self, model_with_layers: model_cls.CreatedModel):
        with pytest.raises(err.CreateError):
            model_with_layers.create()

        model_with_layers.set_outputs(["output"])
        model_with_layers.create()
        assert model_with_layers.built is True


class TestConfigure:
    def test_set_and_get_features(
        self, full_model: model_cls.CreatedModel, csv_str: str
    ):
        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)
        columns = list(dataframe.columns)

        full_model.set_outputs(["output"])
        full_model.create()

        input_features = full_model.get_features("input", "input")
        assert len(input_features) == 0

        with pytest.raises(err.SetError):
            full_model.set_features("input", [], "input")

        with pytest.raises(err.SetError):
            full_model.set_features("input", columns, "input")

        input_columns = columns[:4]
        output_columns = columns[4:]
        full_model.set_features("input", [input_columns[0]], "input")
        input_features = full_model.get_features("input", "input")
        assert len(input_features) == 1
        assert input_features[0] == input_columns[0]
        assert full_model.input_configured is False

        full_model.set_features("input", input_columns, "input")
        input_features = full_model.get_features("input", "input")
        assert len(input_features) == 4
        assert full_model.input_configured is True

        full_model.set_features("output", output_columns, "output")
        output_features = full_model.get_features("output", "output")
        assert len(output_features) == 6
        assert full_model.output_configured is True

    def test_set_get_delete_callbacks(self, full_model: model_cls.CreatedModel):
        full_model.set_callback(
            "EarlyStopping", t.EarlyStoppingParams(min_delta=0, patience=0)
        )
        callback = full_model.get_callback("EarlyStopping")
        assert isinstance(callback, tf.keras.callbacks.EarlyStopping)

        full_model.set_callback("TerminateOnNaN", t.CallbackParams())
        callback = full_model.get_callback("TerminateOnNaN")
        assert isinstance(callback, tf.keras.callbacks.TerminateOnNaN)

        full_model.delete_callback("TerminateOnNaN")
        callback = full_model.get_callback("TerminateOnNaN")
        assert callback is None

        callback = full_model.get_callback("EarlyStopping")
        assert isinstance(callback, tf.keras.callbacks.EarlyStopping)


class TestCompile:
    def test_set_and_get_optimizer(self, full_model: model_cls.CreatedModel):
        with pytest.raises(err.SetError):
            full_model.set_optimizer("SomeOptimizer", t.OptimizerParams())
        assert full_model.get_optimizer() is None

        adam_params = t.AdamParams(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        full_model.set_optimizer("Adam", adam_params)
        optimizer = full_model.get_optimizer()
        assert optimizer == "Adam"

        rms_params = t.RMSpropParams(learning_rate=0.001, rho=0.9, momentum=0.0)
        full_model.set_optimizer("RMSprop", rms_params)
        optimizer = full_model.get_optimizer()
        assert optimizer == "RMSprop"

        sgd_params = t.SGDParams(learning_rate=0.01, momentum=0.0)
        full_model.set_optimizer("SGD", sgd_params)
        optimizer = full_model.get_optimizer()
        assert optimizer == "SGD"

    def test_set_and_get_loss(self, full_model: model_cls.CreatedModel):
        assert full_model.get_loss("output") is None

        losses = enums.losses.classes
        for loss in losses:
            full_model.set_loss("output", loss)
            received_loss = full_model.get_loss("output")
            assert received_loss == loss

    def test_set_and_get_metrics(self, full_model: model_cls.CreatedModel):
        metrics = enums.metrics.classes
        for metric in metrics:
            full_model.set_metrics("output", [metric])
            received_metrics = full_model.get_metrics("output")
            assert received_metrics[0] == metric

        full_model.set_metrics("output", metrics)
        received_metrics = full_model.get_metrics("output")
        assert all(received_metrics[i] == metrics[i] for i in range(len(metrics)))

    def test_compile_model(self, model_with_layers: model_cls.CreatedModel):
        with pytest.raises(err.ModelError):
            model_with_layers.compile()

        model_with_layers.set_outputs(["output"])
        model_with_layers.create()

        with pytest.raises(err.ModelError):
            model_with_layers.compile()

        adam_params = t.AdamParams(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model_with_layers.set_optimizer("Adam", adam_params)
        model_with_layers.set_loss("output", "MeanAbsoluteError")
        model_with_layers.set_metrics("output", ["MeanAbsoluteError"])

        model_with_layers.compile()
        assert model_with_layers.compiled is True


class TestTrain:
    def test_fit_model(
        self,
        compiled_model: model_cls.CreatedModel,
        csv_str: str,
        csv_with_diff_types: str,
    ):
        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        columns = list(dataframe.columns)
        input_columns = columns[:4]
        output_columns = columns[4:]
        compiled_model.set_features("input", input_columns, "input")
        compiled_model.set_features("output", output_columns, "output")

        batch_size = 4
        num_epochs = 5
        val_split = 0.1
        compiled_model.fit(dataframe, batch_size, num_epochs, val_split)

        invalid_csv_file = bytes(csv_with_diff_types, "utf-8")
        with io.BytesIO(invalid_csv_file) as buff:
            invalid_dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        with pytest.raises(err.ModelError):
            compiled_model.fit(invalid_dataframe, batch_size, num_epochs, val_split)

    def test_plot_history(self, compiled_model: model_cls.CreatedModel, csv_str: str):
        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        columns = list(dataframe.columns)
        input_columns = columns[:4]
        output_columns = columns[4:]
        compiled_model.set_features("input", input_columns, "input")
        compiled_model.set_features("output", output_columns, "output")

        batch_size = 4
        num_epochs = 5
        val_split = 0.1
        compiled_model.fit(dataframe, batch_size, num_epochs, val_split)

        history = compiled_model.history
        chart = compiled_model.plot_history(list(history.columns.drop("epoch")), True)
        assert isinstance(chart, alt.Chart)


class TestEvaluate:
    def test_evaluate_model(
        self,
        uploaded_model: model_cls.UploadedModel,
        compiled_model: model_cls.CreatedModel,
        csv_str: str,
        csv_with_diff_types: str,
    ):
        h5_file = compiled_model.as_bytes
        with io.BytesIO(h5_file) as buff:
            uploaded_model.upload(buff)

        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        columns = list(dataframe.columns)
        input_columns = columns[:4]
        output_columns = columns[4:]
        uploaded_model.set_features("input", input_columns, "input")
        uploaded_model.set_features("output", output_columns, "output")
        adam_params = t.AdamParams(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        uploaded_model.set_optimizer("Adam", adam_params)
        uploaded_model.set_loss("output", "MeanAbsoluteError")
        uploaded_model.set_metrics("output", ["MeanAbsoluteError"])
        uploaded_model.compile()

        batch_size = 4
        uploaded_model.evaluate(dataframe, batch_size)

        invalid_csv_file = bytes(csv_with_diff_types, "utf-8")
        with io.BytesIO(invalid_csv_file) as buff:
            invalid_dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        with pytest.raises(err.ModelError):
            uploaded_model.evaluate(invalid_dataframe, batch_size)


class TestPredict:
    def test_make_predictions(
        self,
        uploaded_model: model_cls.UploadedModel,
        compiled_model: model_cls.CreatedModel,
        csv_str: str,
        csv_with_diff_types: str,
    ):
        h5_file = compiled_model.as_bytes
        with io.BytesIO(h5_file) as buff:
            uploaded_model.upload(buff)

        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        columns = list(dataframe.columns)
        input_columns = columns[:4]
        output_columns = columns[4:]
        uploaded_model.set_features("input", input_columns, "input")
        uploaded_model.set_features("output", output_columns, "output")

        batch_size = 4
        uploaded_model.predict(dataframe, batch_size)

        invalid_csv_file = bytes(csv_with_diff_types, "utf-8")
        with io.BytesIO(invalid_csv_file) as buff:
            invalid_dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        with pytest.raises(err.ModelError):
            uploaded_model.predict(invalid_dataframe, batch_size)


class TestModel:
    def test_model_info(self, model_with_layers: model_cls.Model, csv_str: str):
        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        assert model_with_layers.built is False
        assert model_with_layers.input_configured is False
        assert model_with_layers.output_configured is False
        assert model_with_layers.built is False
        assert model_with_layers.compiled is False

        model_with_layers.set_outputs(["output"])
        model_with_layers.create()
        assert model_with_layers.built is True

        columns = list(dataframe.columns)
        input_columns = columns[:4]
        output_columns = columns[4:]
        model_with_layers.set_features("input", input_columns, "input")
        model_with_layers.set_features("output", output_columns, "output")
        assert model_with_layers.input_configured is True
        assert model_with_layers.output_configured is True

        adam_params = t.AdamParams(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model_with_layers.set_optimizer("Adam", adam_params)
        model_with_layers.set_loss("output", "MeanAbsoluteError")
        model_with_layers.set_metrics("output", ["MeanAbsoluteError"])
        model_with_layers.compile()
        assert model_with_layers.compiled is True

    def test_reset_state(self, compiled_model: model_cls.Model, csv_str: str):
        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        columns = list(dataframe.columns)
        input_columns = columns[:4]
        output_columns = columns[4:]
        compiled_model.set_features("input", input_columns, "input")
        compiled_model.set_features("output", output_columns, "output")

        compiled_model.reset_state()
        assert compiled_model.built is False
        assert compiled_model.input_configured is False
        assert compiled_model.output_configured is False
        assert compiled_model.compiled is False
