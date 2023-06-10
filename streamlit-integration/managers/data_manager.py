import collections as clns
import io
import itertools as it
from typing import Literal

import data_classes.data as data_cls
import data_classes.model as model_cls
import pandas as pd
import plotly as ply
import plotly.express as px

import managers.errors as err


def file_exists(data: data_cls.Data) -> bool:
    """Check if data file exists.

    Parameters
    ----------
    data : Data
        Data container object.

    Returns
    -------
    bool
        True if file exists, False otherwise.
    """
    return False if data.file.empty else True


def upload_file(
    buff: io.BytesIO | None, data: data_cls.Data, model: model_cls.Model
) -> None:
    """Read file to pandas format.

    Parameters
    ----------
    buff : File-like object or None
        Buffer object to upload.
    data : Data
        Data container object.
    model : Model
        Model container object.

    Raises
    ------
    UploadError
        For errors with uploading procedure.
    """
    if not buff:
        return

    try:
        data.file = pd.read_csv(buff, header=0, skipinitialspace=True)
        refresh_data(data)
    except ValueError as error:
        raise err.UploadError(f"Unable to upload the file!") from error


def refresh_data(data: data_cls.Data) -> None:
    """Refresh attributes of data container.

    Parameters
    ----------
    data : Data
        Data container object.
    """
    data.columns = list(data.file.columns)
    data.columns_counter = clns.Counter(data.columns)
    data.available_columns = data.columns.copy()


def show_data_stats(data: data_cls.Data) -> pd.DataFrame | None:
    """Show summary of data statistics.

    Parameters
    ----------
    data : Data
        Data container object.

    Returns
    -------
    DataFrame or None
        Dataframe with data statistics. None if data file is absent.
    """
    if not file_exists(data):
        return

    data_stats = pd.concat(
        [
            data.file.describe().transpose(),
            data.file.dtypes.rename("type"),
            pd.Series(data.file.isnull().mean().round(3).mul(100), name="% of NULLs"),
        ],
        axis=1,
    )

    return data_stats


def show_data_plot(x: str, y: str, data: data_cls.Data) -> ply.graph_objs.Figure:
    """Show data plot.

    Parameters
    ----------
    x : str
        X-axis column name.
    y : str
        Y-axis column name.
    data : Data
        Data container object.

    Returns
    -------
    Figure
        Plotly figure object.
    """
    if x == y:
        fig_data = data.file.loc[:, x]
        fig = px.histogram(fig_data, x=x)
    else:
        fig_data = data.file.loc[:, [x, y]].sort_values(by=x)
        fig = px.line(fig_data, x=x, y=y)

    return fig


def get_layer_columns(
    layer_type: Literal["Input", "Output"], layer: str, data: data_cls.Data
) -> list[str]:
    """Get columns attached to the layer.

    Parameters
    ----------
    layer_type : "Input" or "Output"
        Type of the layer.
    layer : str
        Name of the layer.
    data : Data
        Data container object.

    Returns
    -------
    list of str
        List of columns' names. If data container did not contain the layer, the list
        will be empty.
    """
    if layer_type == "Input":
        data_columns = data.input_columns
    else:
        data_columns = data.output_columns

    if not data_columns.get(layer):
        data_columns[layer] = list()

    return data_columns[layer]


def set_columns(
    layer_type: Literal["Input", "Output"],
    layer: str,
    columns: list[str],
    data: data_cls.Data,
    model: model_cls.Model,
) -> None:
    """Set columns for the layer.

    Parameters
    ----------
    layer_type : "Input" or "Output"
        Type of the layer.
    layer : str
        Name of the layer.
    columns : list of str
        List of columns' names to set.
    data : Data
        Data container object.
    model : Model
        Model container object.

    Raises
    ------
    NoColumnsSelectedError
        When no columns are selected.
    LayerOverfilledError
        When the addition of new columns exceeds the layer's capacity.
    """
    if not columns:
        raise err.NoColumnsSelectedError("Please, select at least one column!")

    if layer_type == "Input":
        columns_type = data.input_columns
        shape = model.input_shapes
    else:
        columns_type = data.output_columns
        shape = model.output_shapes

    if len(columns) > shape[layer]:
        raise err.LayerOverfilledError(
            "Please, do not select more columns than the layer's shape allows!"
        )

    # Use clns.Counter to count the reserved columns, then compare them with
    # all the data columns to determine the available ones. Here, it.chain.from_iterable
    # is used to unpack the lists of columns for each layer.
    columns_type[layer] = columns
    input_counter = clns.Counter(it.chain.from_iterable(data.input_columns.values()))
    output_counter = clns.Counter(it.chain.from_iterable(data.output_columns.values()))
    selected_counter = input_counter | output_counter
    data.available_columns = [
        item
        for item, count in data.columns_counter.items()
        if count > selected_counter[item]
    ]
