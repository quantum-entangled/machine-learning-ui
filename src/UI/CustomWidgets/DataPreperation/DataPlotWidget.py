from functools import partial
from typing import Any, Protocol

from bqplot import Toolbar
from bqplot import pyplot as plt
from IPython.display import display
from ipywidgets import Button, Dropdown, Output, TwoByTwoLayout, VBox


class File(Protocol):
    """Protocol for data files."""

    file: Any
    headers: Any


class DataPlotWidget(VBox):
    """Widget to display a data plot."""

    show_plot_button = Button(description="Show Data Plot")
    plot_output = Output()

    widget_children = [show_plot_button, plot_output]

    def __init__(self, data_file: File, **kwargs) -> None:
        """Initialize the data plot widget window."""
        self.show_plot_button.on_click(partial(show_plot, data_file=data_file))

        super().__init__(children=self.widget_children, **kwargs)


@DataPlotWidget.plot_output.capture(clear_output=True, wait=True)
def show_plot(*args, data_file: File) -> None:
    """Show plot of the given file features."""
    if data_file.file is not None:
        headers = data_file.headers
        dropdown_options = [(header, pos) for pos, header in enumerate(headers)]
        x_dropdown = Dropdown(description="x", options=dropdown_options, value=0)
        y_dropdown = Dropdown(description="y", options=dropdown_options, value=0)

        fig = plt.figure()
        plt.plot(
            data_file.file[:, x_dropdown.value],
            data_file.file[:, y_dropdown.value],
            figure=fig,
        )
        plt.xlabel(headers[x_dropdown.value])
        plt.ylabel(headers[y_dropdown.value])

        def on_dropdown_value_change(*args):
            plt.current_figure().marks[0].x = data_file.file[:, x_dropdown.value]
            plt.current_figure().marks[0].y = data_file.file[:, y_dropdown.value]
            plt.xlabel(headers[x_dropdown.value])
            plt.ylabel(headers[y_dropdown.value])

        x_dropdown.observe(on_dropdown_value_change, names="value")
        y_dropdown.observe(on_dropdown_value_change, names="value")

        plot_window = TwoByTwoLayout(
            top_left=VBox([x_dropdown, y_dropdown]),
            top_right=VBox([fig, Toolbar(figure=fig)]),
            align_items="center",
            height="auto",
            width="auto",
        )

        display(plot_window)
    else:
        print("Please, upload the file first!\u274C")
