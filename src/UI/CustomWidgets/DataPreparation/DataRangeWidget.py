from typing import Any, Protocol

import ipywidgets as iw



class DataManager(Protocol):
    """Protocol for data managers."""

    def set_range_grid(self, begin_r:Protocol,end_r:Protocol,
                       begin_c:Protocol,end_c:Protocol, output_handler: Any) -> None:
        ...


class DataRangeWidget(iw.VBox):
    """Widget to set a data grid."""

    name = "Set Data Grid"

    def __init__(self, data_manager: DataManager, **kwargs) -> None:
        """Initialize the data grid widget window."""
        self.data_manager = data_manager

        self._row_range=iw.IntRangeSlider(
            value=[0,10],
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

        super().__init__(children=[self._row_range,self._col_range,self.set_grid_button, self.set_grid_output], **kwargs)

    def _on_set_grid_button_clicked(self, _) -> None:
        self.data_manager.set_range_grid(self._row_range.value[0],self._row_range.value[1],
                                    self._col_range.value[0],self._col_range.value[1],
             output_handler=self.set_grid_output
        )
