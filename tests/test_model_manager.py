from mlui.data_classes import model as model_cls
from mlui.managers import model_manager as mm


def test_model_exists(model: model_cls.Model) -> None:
    exists = mm.model_exists(model)

    assert exists is False
