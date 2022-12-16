import pytest
from src.Managers.ModelManager import ModelManager


@pytest.fixture
def model_manager(data, model) -> ModelManager:
    return ModelManager(data=data, model=model)


def test_create_model(model_manager: ModelManager) -> None:
    model_manager.create_model(model_name="test_create_model")

    assert model_manager.model_instance is not None


def test_upload_model(model_manager: ModelManager) -> None:
    model_manager.upload_model("db/Models/test_model.h5")

    assert model_manager.model_instance is not None


def test_refresh_model(model_manager: ModelManager) -> None:
    model_manager.create_model(model_name="test_refresh_model")
    model_manager.refresh_model()

    assert model_manager.name == "test_refresh_model"
    assert model_manager.input_layers == dict()
    assert model_manager.output_layers == dict()
    assert model_manager.layers == dict()
    assert model_manager.input_shapes == dict()
    assert model_manager.output_shapes == dict()
    assert model_manager.losses == dict()
    assert model_manager.metrics == dict()
