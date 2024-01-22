import mlui.types.classes as t
from mlui.classes import errors


def validate_shapes(shapes: t.Shapes) -> None:
    """
    Validate the shapes of a model.

    Parameters
    ----------
    shapes : dict of tuples, list of tuples or tuple
        Shapes to be validated. Single shape is a `tuple` of `(None, int)`.

    Raises
    ------
    ValidateModelError
        If the shapes are empty. If any individual shape is empty. If any individual
        shape contains more than 2 dimensions.
    """
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
