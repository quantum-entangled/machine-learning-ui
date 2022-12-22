from typing import Protocol

import ipywidgets as iw
import numpy as np

from src.Enums.ErrorMessages import Error


class DataManager(Protocol):
    """Protocol for data managers."""

    def file_exists(self) -> bool:
        ...

    @property
    def input_train_data(self) -> dict[str, np.ndarray]:
        ...

    @property
    def output_train_data(self) -> dict[str, np.ndarray]:
        ...


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def fit_model(self, batch_size: int, num_epochs: int, val_split: float) -> None:
        ...

    @property
    def compiled(self) -> bool:
        ...


class TrainModelWidget(iw.VBox):

    name = "Train Model"

    def __init__(
        self, data_manager: DataManager, model_manager: ModelManager, **kwargs
    ):
        """Initialize widget window."""
        # Managers
        self.data_manager = data_manager
        self.model_manager = model_manager

        # Widgets
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
        self.val_split = iw.BoundedFloatText(
            value=0.15,
            min=0.01,
            max=1,
            step=0.01,
            description="Validation split:",
            style={"description_width": "initial"},
        )
        self.train_model_button = iw.Button(description="Train Model")
        self.train_output = iw.Output()

        # Callbacks
        self.train_model_button.on_click(self._on_train_model_button_clicked)

        super().__init__(
            children=[
                self.batch_size,
                self.num_epochs,
                self.val_split,
                self.train_model_button,
                self.train_output,
            ]
        )

    def _on_train_model_button_clicked(self, _) -> None:
        """Callback for train model button."""
        self.train_output.clear_output(wait=True)

        with self.train_output:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            if not (
                self.data_manager.input_train_data
                and self.data_manager.output_train_data
            ):
                print(Error.DATA_NOT_SPLIT)
                return

            if not self.model_manager.compiled:
                print(Error.MODEL_NOT_COMPILED)
                return

            batch_size = self.batch_size.value
            num_epochs = self.num_epochs.value
            val_split = self.val_split.value

            self.model_manager.fit_model(
                batch_size=batch_size,
                num_epochs=num_epochs,
                val_split=val_split,
            )

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.train_output.clear_output()

    def _on_file_uploaded(self) -> None:
        """Callback for file upload."""
        self.train_output.clear_output()

    def _on_data_split(self) -> None:
        """Callback for data split."""
        self.train_output.clear_output()

    def _on_model_compiled(self) -> None:
        """Callback for model compilation."""
        self.train_output.clear_output()
