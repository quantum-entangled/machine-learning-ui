from enum import Enum


class Success(Enum):
    FILE_UPLOADED = "File has been successfully uploaded!\u2705"
    MODEL_CREATED = "Model has been successfully created!\u2705"
    MODEL_UPLOADED = "Model has been successfully uploaded!\u2705"
    LAYER_ADDED = "Layer has been successfully added!\u2705"
    MODEL_SAVED = "Model has been successfully saved!\u2705"
    COLUMNS_ADDED = "Columns have been successfully added!\u2705"
    CALLBACK_ADDED = "Callback has been successfully added!\u2705"
    LOSS_ADDED = "Loss function has been successfully added!\u2705"
    METRIC_ADDED = "Metric has been successfully added!\u2705"
    OPTIMIZER_SELECTED = "Optimizer has been successfully selected!\u2705"
    OUTPUTS_SET = "Model outputs have been successfully set!\u2705"
    MODEL_COMPILED = "Model has been successfully compiled!\u2705"

    def __str__(self):
        return f"{self.value}"
