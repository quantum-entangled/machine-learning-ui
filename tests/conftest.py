import pytest

from mlui.data_classes import data as data_cls
from mlui.data_classes import model as model_cls


@pytest.fixture
def data() -> data_cls.Data:
    return data_cls.Data()


@pytest.fixture
def model() -> model_cls.Model:
    return model_cls.Model()
