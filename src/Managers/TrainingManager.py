from typing import Any, Protocol


class Data(Protocol):
    """Protocol for data files."""

    file: Any
    headers: list[Any]


class Model(Protocol):
    """Protocol for models."""

    name: str | None
    instance: Any
    layers: dict[Any, Any]
    layers_shapes: dict[Any, Any]
    input_names: list[str]
    output_names: list[str]


class Config(Protocol):
    """Protocol for configs."""

    num_columns_per_layer: dict[str, int]
    input_training_columns: dict[Any, Any]
    output_training_columns: dict[Any, Any]
    optimizer: Any
    losses: dict[Any, Any]
    metrics: dict[Any, Any]
    callbacks: list[Any]
    train_history: Any


class DataManager(Protocol):
    """Protocol for data managers."""

    @property
    def data(self) -> Data:
        ...

    def upload_file(self, file_chooser: Any, output_handler: Any) -> None:
        ...


class ModelManager(Protocol):
    """Protocol for model managers."""

    @property
    def model(self) -> Model:
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

    def set_num_columns_per_layer(self, layer_name: str) -> None:
        if layer_name not in self._config.num_columns_per_layer.keys():
            self._config.num_columns_per_layer.update({layer_name: 0})

    def update_num_columns_per_layer(self, layer_name: str, num_columns: int) -> None:
        self._config.num_columns_per_layer[layer_name] += num_columns

    def add_training_columns(
        self, layer_type: str, layer_name: str, from_column: Any, to_column: Any
    ) -> None:
        if layer_type == "input":
            if layer_name not in self._config.input_training_columns.keys():
                self._config.input_training_columns[layer_name] = list()

            self._config.input_training_columns[layer_name] = sorted(
                set(
                    self._config.input_training_columns[layer_name]
                    + list(range(from_column, to_column))
                )
            )
        else:
            if layer_name not in self._config.output_training_columns.keys():
                self._config.output_training_columns[layer_name] = list()

            self._config.output_training_columns[layer_name] = sorted(
                set(
                    self._config.output_training_columns[layer_name]
                    + list(range(from_column, to_column))
                )
            )

    def select_optimizer(self, instance: Any, **kwargs) -> None:
        self._config.optimizer = instance(**kwargs)

    def add_loss(self, layer_name: str, loss: Any) -> None:
        self._config.losses.update({layer_name: loss})

    def add_metric(self, layer_name: str, metric: Any) -> None:
        self._config.metrics.update({layer_name: metric})

    def add_callback(self, instance: Any, **kwargs) -> None:
        self._config.callbacks += [instance(**kwargs)]

    def compile_model(self) -> None:
        self._model_manager._model.instance.compile(
            optimizer=self._config.optimizer,
            loss=self._config.losses,
            metrics=self._config.metrics,
        )

    def check_shapes(self) -> str:
        for layer_name in list(
            self._config.input_training_columns | self._config.output_training_columns
        ):
            if (
                self._config.num_columns_per_layer[layer_name]
                < self._model_manager.model.layers_shapes[layer_name]
            ):
                return layer_name

        return str()

    def fit_model(
        self, batch_size: int, num_epochs: int, validation_split: float
    ) -> None:
        self._config.train_history = self._model_manager.model.instance.fit(
            x={
                layer_name: self._data_manager.data.file[
                    :, self._config.input_training_columns[layer_name]
                ]
                for layer_name in self._config.input_training_columns.keys()
            },
            y={
                layer_name: self._data_manager.data.file[
                    :, self._config.output_training_columns[layer_name]
                ]
                for layer_name in self._config.output_training_columns.keys()
            },
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=validation_split,
            callbacks=self._config.callbacks,
            verbose=1,
        )

    @property
    def data(self) -> Data:
        return self._data_manager.data

    @property
    def model(self) -> Model:
        return self._model_manager.model

    @property
    def config(self) -> Config:
        return self._config
