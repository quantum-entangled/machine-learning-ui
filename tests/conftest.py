import pytest
from src.DataClasses import Data, Model


@pytest.fixture
def data() -> Data:
    return Data()


@pytest.fixture
def model() -> Model:
    return Model()
