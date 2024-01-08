import mlui.types.classes as t
from mlui.classes import errors


def validate_shapes(model: t.Object, at: t.Side) -> None:
    if at == "input":
        shapes = model.input_shape
    else:
        shapes = model.output_shape

    if not shapes:
        raise errors.ValidateModelError("The model's shapes are empty!")

    if isinstance(shapes, dict):
        shapes = list(shapes.values())
    elif isinstance(shapes, tuple):
        shapes = [shapes]

    for shape in shapes:
        if not shapes:
            raise errors.ValidateModelError(
                "At least one of the model's shapes is empty!"
            )

        if len(shape) > 2:
            raise errors.ValidateModelError(
                "At least one of the model's shapes contains more than 2 dimensions!"
            )
