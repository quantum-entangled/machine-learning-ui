class UploadError(Exception):
    """For errors during the uploading of files/models."""


class IncorrectFileStructure(Exception):
    """When the file structure is incorrect."""


class FileEmptyError(Exception):
    """When trying to upload an empty file."""


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


class IncorrectTestDataPercentage(Exception):
    """When the test data percentage is incorrect."""


class NoOptimizerError(Exception):
    """When no optimizer is set for the model."""


class NoLossError(Exception):
    """When loss functions are not set for each layer."""


class SameCallbackError(Exception):
    """When trying to set the already attached callback to the model."""


class DataNotSplitError(Exception):
    """When the dataset is not split into training and testing sets."""


class ModelNotCompiledError(Exception):
    """When the model is not compiled."""


class ModelNotTrainedError(Exception):
    """When the model is not trained."""
