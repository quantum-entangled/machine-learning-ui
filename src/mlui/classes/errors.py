class UploadError(Exception):
    """For errors during the uploading of files/models."""


class CreateError(Exception):
    """For errors during the process of creating a model."""


class ParseCSVError(Exception):
    """For errors during the process of parsing a file."""


class ValidateDataError(Exception):
    """For errors during the process of validating data."""


class ModelError(Exception):
    """For arbitrary errors during the process of executing the model's methods."""


class LayerError(Exception):
    """For errors during the process of constructing a layer."""


class SetError(Exception):
    """For arbitrary errors during the process of setting an attribute's value."""


class DeleteError(Exception):
    """For arbitrary errors during the process of deleting an attribute's value."""


class PlotError(Exception):
    """For errors during the process of displaying a plot."""
