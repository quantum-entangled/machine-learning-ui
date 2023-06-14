class UploadError(Exception):
    """For errors during the uploading of files/models."""


class NoModelNameError(Exception):
    """When trying to create a model with no name."""


class SameModelNameError(Exception):
    """When trying to create a model with already existing name."""


class NoModelError(Exception):
    """When model is not instantiated."""


class NoLayerNameError(Exception):
    """When trying to create a layer with no name."""


class SameLayerNameError(Exception):
    """When trying to create a layer with already existing name."""


class NoConnectionError(Exception):
    """When no connection is provided for the layer."""


class NoOutputsSelectedError(Exception):
    """When no layers are selected for the model outputs."""


class NoOutputLayersError(Exception):
    """When there are no output layers in the model."""


class LayerOverfilledError(Exception):
    """When the addition of new columns exceeds the layer's capacity."""


class NoColumnsSelectedError(Exception):
    """When no columns are selected."""


class InputsUnderfilledError(Exception):
    """When some of the input layers are not filled with the data columns."""


class OutputsUnderfilledError(Exception):
    """When some of the output layers are not filled with the data columns."""


class SameMetricError(Exception):
    """When trying to set the already attached metric to the layer."""