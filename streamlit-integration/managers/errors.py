class UploadError(Exception):
    """Exception raised for errors during uploading of files/models."""


class NoModelNameError(Exception):
    """Exception raised when trying to create a model with no name."""


class SameModelNameError(Exception):
    """Exception raised when trying to create a model with already existing name."""


class NoModelError(Exception):
    """Exception raised when a model doesn't exist."""


class NoLayerNameError(Exception):
    """Exception raised when trying to create a layer with no name."""


class SameLayerNameError(Exception):
    """Exception raised when trying to create a layer with already existing name."""


class NoConnectionError(Exception):
    """Exception raised when no connection is provided for a layer."""
