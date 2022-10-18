from functools import partial
from typing import Any, Protocol

from bqplot import Axis, Figure, LinearScale, Lines
from IPython.display import display
from ipywidgets import Button, Dropdown, Output, TwoByTwoLayout, VBox, link


class File(Protocol):
    """Protocol for data files."""

    file: Any


class DataPlotWidget(VBox):

    show_plot_button = Button(description="Show Data Plot")
    plot_output = Output()

    widget_children = [show_plot_button, plot_output]

    def __init__(self, data_file: File, **kwargs) -> None:

        self.show_plot_button.on_click(partial(show_plot, data_file=data_file))

        super().__init__(children=self.widget_children, **kwargs)


@DataPlotWidget.plot_output.capture(clear_output=True, wait=True)
def show_plot(*args, data_file: File):
    try:
        headers = data_file.headers
        dropdown_options = [(header, pos) for pos, header in enumerate(headers)]

        x_dropdown = Dropdown(description="x", options=dropdown_options, value=0)
        y_dropdown = Dropdown(description="y", options=dropdown_options, value=0)

        def on_dropdown_value_change(*args):
            x_sc = LinearScale()
            y_sc = LinearScale()
            ax_x = Axis(label="x", scale=x_sc)
            ax_y = Axis(label="y", scale=y_sc, orientation="vertical")
            line = Lines(
                x=data_file.file[:, x_dropdown.value],
                y=data_file.file[:, y_dropdown.value],
                scales={"x": x_sc, "y": y_sc},
            )
            fig = Figure(axes=[ax_x, ax_y], marks=[line])
            plot_window.bottom_right = fig

        x_dropdown.observe(on_dropdown_value_change, names="value")
        y_dropdown.observe(on_dropdown_value_change, names="value")

        plot_window = TwoByTwoLayout(
            top_left=x_dropdown,
            bottom_left=y_dropdown,
            align_items="center",
            height="auto",
            width="auto",
        )

        display(plot_window)
    except AttributeError:
        print("Please, upload the file first!\u274C")
