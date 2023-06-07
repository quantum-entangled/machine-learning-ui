class UploadError(Exception):
    """Exception raised for errors during uploading of files/models."""


class NoModelNameError(Exception):
    """Exception raised when trying to create a model with no name."""


class SameModelNameError(Exception):
    """Exception raised when trying to create a model with already existing name."""


class NoModelError(Exception):
    """Exception raised when model isn't instantiated."""


class NoLayerNameError(Exception):
    """Exception raised when trying to create a layer with no name."""


class SameLayerNameError(Exception):
    """Exception raised when trying to create a layer with already existing name."""


class NoConnectionError(Exception):
    """Exception raised when no connection is provided for the layer."""


class NoOutputsSelectedError(Exception):
    """Exception raised when no layers are selected for the model outputs."""


class NoOutputLayersError(Exception):
    """Exception raised when there are no output layers in the model."""
