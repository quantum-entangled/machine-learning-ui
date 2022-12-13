from typing import Protocol

import ipywidgets as iw

from Enums.ErrorMessages import Error


class DataManager(Protocol):
    """Protocol for data managers."""

    def file_exists(self) -> bool:
        ...

    def set_range_grid(self, begin_r: Protocol, end_r: Protocol,
                       begin_c: Protocol, end_c: Protocol) -> None:
        ...

    def show_data_grid(self) -> None:
        ...


class DataGridWidget(iw.VBox):
    """Widget to display a data grid."""

    name = "Show Data Grid"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize the data grid widget window."""
        # Managers
        self.data_manager = data_manager
        # widgets for set range
        self._row_range = iw.IntRangeSlider(
            value=[0, 10],
            min=0,
            max=1000,
            step=5,
            description='Row range',
            disable=False,
            continuous=True,
            orientation='vertical',
            readout=True,
            readout_fomat='d',
        )
        self._col_range = iw.IntRangeSlider(
            value=[0, 10],
            min=0,
            max=1000,
            step=5,
            description='Column range',
            disable=False,
            continuous=True,
            orientation='horizontal',
            readout=True,
            readout_fomat='d',
        )

        self.set_grid_button = iw.Button(description="Set Data Grid")
        self.set_grid_output = iw.Output()
        self.set_grid_button.on_click(self._on_set_grid_button_clicked)

        # Widgets
        self.show_grid_button = iw.Button(description="Show Data Grid")
        self.grid_output = iw.Output()

        # Callbacks
        self.show_grid_button.on_click(self._on_show_grid_button_clicked)

        super().__init__(children=[self._row_range, self._col_range, self.set_grid_button, self.set_grid_output,
                                   self.show_grid_button, self.grid_output])

    def _on_show_grid_button_clicked(self, _) -> None:
        """Callback for show data grid button."""
        self.grid_output.clear_output(wait=True)

        with self.grid_output:
            if not self.data_manager.file_exists():
                print(Error.NO_FILE_UPLOADED)
                return

            self.data_manager.show_data_grid()

    def _on_widget_state_change(self) -> None:
        """Callback for parent widget ensemble."""
        self.grid_output.clear_output()

    def _on_set_grid_button_clicked(self, _) -> None:
        self.data_manager.set_range_grid(self._row_range.value[0], self._row_range.value[1],
                                         self._col_range.value[0], self._col_range.value[1]
                                         )