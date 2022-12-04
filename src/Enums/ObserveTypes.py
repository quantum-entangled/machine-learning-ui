from enum import Enum


class Observe(str, Enum):
    FILE = "_on_file_uploaded"
    MODEL = "_on_model_instantiated"
    LAYER_ADDED = "_on_layer_added"
    OUTPUTS_SET = "_on_outputs_set"
    OPTIMIZER_SELECTED = "_on_optimizer_selected"
    LOSSES_SELECTED = "_on_losses_selected"
    INPUT_COLUMNS_ADDED = "_on_input_columns_added"
    OUTPUT_COLUMNS_ADDED = "_on_output_columns_added"
    LAYERS_FILLED = "_on_layers_filled"
    DATA_SPLIT = "_on_data_split"
    MODEL_COMPILED = "_on_model_compiled"
    MODEL_TRAINED = "_on_model_trained"
