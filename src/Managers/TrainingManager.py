from typing import Any, Protocol


class Config(Protocol):
    """Protocol for configs."""

    input_training_indices: Any
    output_training_indices: Any
    optimizer: Any


class DataManager(Protocol):
    """Protocol for data managers."""

    @property
    def data(self) -> Any:
        ...

    def upload_file(self, file_chooser: Any, output_handler: Any) -> None:
        ...


class ModelManager(Protocol):
    """Protocol for model managers."""

    @property
    def model(self) -> Any:
        ...

    def upload_model(self, file_chooser: Any, output_handler: Any) -> None:
        ...


class TrainingManager:
    def __init__(
        self, config: Config, data_manager: DataManager, model_manager: ModelManager
    ) -> None:
        """Initialize the internal data and model managers."""
        self._config = config
        self._data_manager = data_manager
        self._model_manager = model_manager

    def upload_file(self, file_chooser: Any, output_handler: Any) -> None:
        self._data_manager.upload_file(
            file_chooser=file_chooser, output_handler=output_handler
        )

    def upload_model(self, file_chooser: Any, output_handler: Any) -> None:
        self._model_manager.upload_model(
            file_chooser=file_chooser, output_handler=output_handler
        )

    def check_instances(self, output_handler: Any) -> bool:
        output_handler.clear_output(wait=True)

        if (
            self._data_manager.data.file is None
            or self._model_manager.model.instance is None
        ):
            with output_handler:
                print("Please, upload the model and/or data first!\u274C")
                return False

        return True

    def add_training_columns(
        self, layer_type: str, layer_name: str, indices: Any
    ) -> None:
        inp_ind = self._config.input_training_indices
        out_ind = self._config.output_training_indices

        if layer_type == "input":
            if layer_name not in inp_ind.keys():
                inp_ind[layer_name] = list()

            inp_ind[layer_name] = sorted(inp_ind[layer_name] + indices)
        else:
            if layer_name not in out_ind.keys():
                out_ind[layer_name] = list()

            out_ind[layer_name] = sorted(out_ind[layer_name] + indices)

    def is_model(self) -> bool:
        return True if self._model_manager.model.instance else False

    def select_optimizer(self, instance: Any, **kwargs) -> None:
        self._config.optimizer = instance(**kwargs)

    @property
    def data(self) -> Any:
        return self._data_manager.data

    @property
    def model(self) -> Any:
        return self._model_manager.model

    @property
    def config(self) -> Config:
        return self._config
