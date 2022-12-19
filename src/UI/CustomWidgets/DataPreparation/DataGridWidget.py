from typing import Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error


class DataManager(Protocol):
    """Protocol for data managers."""

    def file_exists(self) -> bool:
        ...

    def show_data_grid(
        self, begin_row: int, end_row: int, begin_col: int, end_col: int
    ) -> None:
        ...


class DataGridWidget(iw.VBox):
    """Widget to display a data grid."""

    name = "Show Data Grid"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize the data grid widget window."""
        # Managers
        self.data_manager = data_manager

        # Widgets
        self.row_range = iw.IntRangeSlider(
            value=[0, 10],
            min=0,
            max=1000,
            step=5,
            description="Row range:",
            style={"description_width": "initial"},
        )
        self.col_range = iw.IntRangeSlider(
            value=[0, 10],
            min=0,
            max=100,
            step=5,
            description="Column range:",
            style={"description_width": "initial"},
        )
        self.show_grid_button = iw.Button(description="Show Data Grid")
        self.grid_output = iw.Output()

        # Callbacks
        self.show_grid_button.on_click(self._on_show_grid_button_clicked)

        super().__init__(
            children=[
                self.row_range,
                self.col_range,
                self.show_grid_button,
                self.grid_output,
            ]
        )

    def _on_show_grid_button_clicked(self, _) -> None:
        """Callback for show data grid button."""
        self.grid_output.clear_output(wait=True)

        with self.grid_output:
            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            self.data_manager.show_data_grid(
                begin_row=self.row_range.value[0],
                end_row=self.row_range.value[1],
                begin_col=self.col_range.value[0],
                end_col=self.col_range.value[1],
            )

    def _on_file_uploaded(self) -> None:
        """Callback for file upload."""
        self.grid_output.clear_output()
