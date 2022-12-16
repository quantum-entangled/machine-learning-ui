from typing import Protocol

import ipywidgets as iw

from src.Enums.ErrorMessages import Error


class DataManager(Protocol):
    """Protocol for data managers."""

    def file_exists(self) -> bool:
        ...

    def show_data_stats(self) -> None:
        ...


class DataStatsWidget(iw.VBox):
    """Widget to display a data statistics."""

    name = "Show Data Statistics"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize the data grid widget window."""
        # Managers
        self.data_manager = data_manager

        # Widgets
        self.show_stat_button = iw.Button(description="Show Data Statistics")
        self.stat_output = iw.Output()

        # Callbacks
        self.show_stat_button.on_click(self._on_show_grid_button_clicked)

        super().__init__(children=[self.show_stat_button, self.stat_output])

    def _on_show_grid_button_clicked(self, _) -> None:
        """Callback for show data stat button."""
        self.stat_output.clear_output(wait=True)

        with self.stat_output:
            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            self.data_manager.show_data_stats()

    def _on_file_uploaded(self) -> None:
        """Callback for file upload."""
        self.stat_output.clear_output()
