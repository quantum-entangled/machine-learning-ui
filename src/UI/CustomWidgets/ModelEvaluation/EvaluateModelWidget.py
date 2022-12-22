from typing import Protocol

import ipywidgets as iw
import numpy as np

from src.Enums.ErrorMessages import Error


class DataManager(Protocol):
    """Protocol for data managers."""

    def file_exists(self) -> bool:
        ...

    @property
    def input_test_data(self) -> dict[str, np.ndarray]:
        ...

    @property
    def output_test_data(self) -> dict[str, np.ndarray]:
        ...


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def evaluate_model(self, batch_size: int) -> None:
        ...

    @property
    def compiled(self) -> bool:
        ...


class EvaluateModelWidget(iw.VBox):

    name = "Evaluate Model"

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
        self.evaluate_model_button = iw.Button(description="Evaluate Model")
        self.evaluate_output = iw.Output()

        # Callbacks
        self.evaluate_model_button.on_click(self._on_evaluate_model_button_clicked)

        super().__init__(
            children=[self.batch_size, self.evaluate_model_button, self.evaluate_output]
        )

    def _on_evaluate_model_button_clicked(self, _) -> None:
        """Callback for evaluate model button."""
        self.evaluate_output.clear_output(wait=True)

        with self.evaluate_output:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            if not (
                self.data_manager.input_test_data and self.data_manager.output_test_data
            ):
                print(Error.DATA_NOT_SPLIT)
                return

            if not self.model_manager.compiled:
                print(Error.MODEL_NOT_COMPILED)
                return

            batch_size = self.batch_size.value

            self.model_manager.evaluate_model(batch_size=batch_size)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.evaluate_output.clear_output()

    def _on_file_uploaded(self) -> None:
        """Callback for file upload."""
        self.evaluate_output.clear_output()

    def _on_data_split(self) -> None:
        """Callback for data split."""
        self.evaluate_output.clear_output()

    def _on_model_compiled(self) -> None:
        """Callback for model compilation."""
        self.evaluate_output.clear_output()
