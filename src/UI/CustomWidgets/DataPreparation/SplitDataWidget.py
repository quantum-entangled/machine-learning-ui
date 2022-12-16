from typing import Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error
from src.Enums.SuccessMessages import Success


class DataManager(Protocol):
    """Protocol for data managers."""

    def file_exists(self) -> bool:
        ...

    def check_inputs_fullness(self) -> bool:
        ...

    def check_outputs_fullness(self) -> bool:
        ...

    def split_data(self, test_size: int) -> None:
        ...


class ModelManager(Protocol):
    """Protocol for model managers."""

    def model_exists(self) -> bool:
        ...


class SplitDataWidget(iw.VBox):

    name = "Split Data"

    def __init__(
        self, data_manager: DataManager, model_manager: ModelManager, **kwargs
    ):
        """Initialize widget window."""
        # Managers
        self.data_manager = data_manager
        self.model_manager = model_manager

        # Widgets
        self.test_size_slider = iw.IntSlider(
            value=25,
            min=0,
            max=100,
            step=1,
            description="Select test data %:",
            style={"description_width": "initial"},
        )
        self.split_data_button = iw.Button(description="Split Data")
        self.split_output = iw.Output()

        # Callbacks
        self.split_data_button.on_click(self._on_split_data_button_clicked)

        super().__init__(
            children=[self.test_size_slider, self.split_data_button, self.split_output]
        )

    def _on_split_data_button_clicked(self, _) -> None:
        """Callback for split data button."""
        self.split_output.clear_output(wait=True)

        with self.split_output:
            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            if not self.model_manager.model_exists():
                print(Error.NO_MODEL)
                return

            if not self.data_manager.check_inputs_fullness():
                print(Error.INPUTS_UNDERFILLED)
                return

            if not self.data_manager.check_outputs_fullness():
                print(Error.OUTPUTS_UNDERFILLED)
                return

            self.data_manager.split_data(test_size=self.test_size_slider.value)

            print(Success.DATA_SPLIT)

    def _on_file_uploaded(self) -> None:
        """Callback for file upload."""
        self.split_output.clear_output()

    def _on_model_instantiated(self) -> None:
        """Callback for model instantiation."""
        self.split_output.clear_output()

    def _on_input_columns_added(self) -> None:
        """Callback for adding input columns."""
        self.split_output.clear_output()

    def _on_output_columns_added(self) -> None:
        """Callback for adding output columns."""
        self.split_output.clear_output()
