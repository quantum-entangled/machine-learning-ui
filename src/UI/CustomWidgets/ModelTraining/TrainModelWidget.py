from typing import Any, Protocol

import ipywidgets as iw


class Manager(Protocol):
    """Protocol for training managers."""

    @property
    def data(self) -> Any:
        ...

    @property
    def model(self) -> Any:
        ...

    @property
    def config(self) -> Any:
        ...

    def compile_model(self) -> None:
        ...

    def check_shapes(self) -> str:
        ...

    def fit_model(
        self, batch_size: int, num_epochs: int, validation_split: float
    ) -> None:
        ...


class TrainModelWidget(iw.VBox):

    name = "Train Model"

    def __init__(self, manager: Manager, **kwargs):
        self._manager = manager

        self.batch_size = iw.BoundedIntText(
            value=32,
            min=1,
            max=512,
            step=1,
            description="Batch size:",
            style={"description_width": "initial"},
        )
        self.num_epochs = iw.BoundedIntText(
            value=30,
            min=1,
            max=200,
            step=1,
            description="Number of epochs:",
            style={"description_width": "initial"},
        )
        self.validation_split = iw.BoundedFloatText(
            value=0.15,
            min=0,
            max=1,
            step=0.01,
            description="Validation split:",
            style={"description_width": "initial"},
        )
        self.train_model_button = iw.Button(description="Train Model")
        self.train_model_button.on_click(self._on_train_model_button_clicked)
        self.train_output = iw.Output()

        super().__init__(
            children=[
                self.batch_size,
                self.num_epochs,
                self.validation_split,
                self.train_model_button,
                self.train_output,
            ],
            **kwargs,
        )

    def _on_train_model_button_clicked(self, _) -> None:
        self.train_output.clear_output(wait=True)

        if self._manager.data.file is None:
            with self.train_output:
                print("Please, upload the file first!\u274C")
            return

        if self._manager.model.instance is None:
            with self.train_output:
                print("Please, upload the model first!\u274C")
            return

        if not (
            self._manager.config.input_training_columns
            and self._manager.config.output_training_columns
        ):
            with self.train_output:
                print("Please, select the training data first!\u274C")
            return

        with self.train_output:
            layer_name = self._manager.check_shapes()

            if layer_name:
                print(
                    f"Layer '{layer_name}' is not fully filled with data columns!\u274C"
                )
                return

        if not self._manager.config.optimizer:
            with self.train_output:
                print("Please, select the optimizer first!\u274C")
            return

        if not self._manager.config.losses:
            with self.train_output:
                print("Please, select the loss function(s) first!\u274C")
            return

        self._manager.compile_model()

        batch_size = self.batch_size.value
        num_epochs = self.num_epochs.value
        validation_split = self.validation_split.value

        with self.train_output:
            self._manager.fit_model(
                batch_size=batch_size,
                num_epochs=num_epochs,
                validation_split=validation_split,
            )
