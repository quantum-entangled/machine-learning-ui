from typing import Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error


class DataManager(Protocol):
    """Protocol for data managers."""

    def file_exists(self) -> bool:
        ...

    def check_inputs_fullness(self) -> bool:
        ...


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...

    def make_predictions(self, batch_size: int) -> None:
        ...

    @property
    def compiled(self) -> bool:
        ...


class MakePredictionsWidget(iw.VBox):

    name = "Make Predictions"

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
        self.predict_button = iw.Button(description="Make Predictions")
        self.predict_output = iw.Output()

        # Callbacks
        self.predict_button.on_click(self._on_predict_button_clicked)

        super().__init__(
            children=[self.batch_size, self.predict_button, self.predict_output]
        )

    def _on_predict_button_clicked(self, _) -> None:
        """Callback for evaluate model button."""
        self.predict_output.clear_output(wait=True)

        with self.predict_output:
            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            if not self.data_manager.check_inputs_fullness():
                print(Error.INPUTS_UNDERFILLED)
                return

            if not self.model_manager.compiled:
                print(Error.MODEL_NOT_COMPILED)
                return

            batch_size = self.batch_size.value

            self.model_manager.make_predictions(batch_size=batch_size)

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.predict_output.clear_output()

    def _on_file_uploaded(self) -> None:
        """Callback for file upload."""
        self.predict_output.clear_output()

    def _on_input_columns_added(self) -> None:
        """Callback for adding input columns."""
        self.predict_output.clear_output()

    def _on_model_compiled(self) -> None:
        """Callback for model compilation."""
        self.predict_output.clear_output()
