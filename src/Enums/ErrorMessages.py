from enum import Enum


class Error(Enum):
    NO_FILE_PATH = "Please, select the file first!\u274C"
    NO_FILE_UPLOADED = "Please, upload the file first!\u274C"
    DATA_NOT_SPLIT = "Please, split the data first!\u274C"
    NO_MODEL = "Please, create or upload the model first!\u274C"
    NO_MODEL_PATH = "Please, select the model first!\u274C"
    NO_MODEL_NAME = "Please, enter the model name first!\u274C"
    SAME_MODEL_NAME = (
        "Model with this name already exists. Please, enter a different name!\u274C"
    )
    NO_MODEL_OUTPUTS = "Please, select the model outputs first!\u274C"
    MODEL_NOT_COMPILED = "Please, compile the model first!\u274C"
    MODEL_NOT_TRAINED = "Please, train the model first!\u274C"
    NO_LAYERS = "Please, select the layer(s) first!\u274C"
    NO_LAYER_NAME = "Please, enter the layer name first!\u274C"
    NO_OUTPUT_LAYERS = "There are no output layers in the model!\u274C"
    SAME_LAYER_NAME = (
        "Layer with this name already exists. Please, enter a different name!\u274C"
    )
    LAYER_OVERFILLED = "You've selected more columns than the layer can accept!\u274C"
    NO_COLUMNS_SELECTED = "Please, select the data columns first!\u274C"
    NO_CONNECT_TO = "Please, select the layer(s) to connect!\u274C"
    INPUTS_UNDERFILLED = "Please, fill all input layers with data columns first!\u274C"
    OUTPUTS_UNDERFILLED = (
        "Please, fill all output layers with data columns first!\u274C"
    )
    NO_OPTIMIZER = "Please, select the model optimizer first!\u274C"
    NO_LOSS = "Please, select the loss function for each output first!\u274C"
    SAME_METRIC = "Metric has already been added to this layer!\u274C"
    SAME_CALLBACK = "Callback has already been added!\u274C"

    def __str__(self):
        return f"{self.value}"
