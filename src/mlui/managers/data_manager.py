import collections as clns
import csv
import io
import itertools as it
from typing import Literal

import pandas as pd
import plotly as ply
import plotly.express as px
import sklearn.model_selection as sk

import mlui.data_classes.data as data_cls
import mlui.data_classes.model as model_cls
import mlui.managers.errors as err


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


def upload_file(buff: io.BytesIO, data: data_cls.Data) -> None:
    """Read a file to the pandas format.

    Parameters
    ----------
    buff : File-like object
        Buffer object to upload.
    data : Data
        Data container object.

    Raises
    ------
    ParsingFileError
        When the uploaded file has structure inconsistencies.
    FileEmptyError
        When trying to upload an empty file.
    UploadError
        For errors with uploading procedure.
    """
    csv_file = buff.read().decode("utf-8")
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(csv_file).delimiter
    has_header = sniffer.has_header(csv_file)

    if delimiter not in (",", ";"):
        raise err.ParsingFileError(
            "The file doesn't contain the expected delimiter (',' or ';')"
        )

    if not has_header:
        raise err.ParsingFileError("The file doesn't contain a header!")

    reader = csv.reader(csv_file, delimiter=delimiter, skipinitialspace=True)
    header_len = len(next(reader))

    if header_len < 2:
        raise err.ParsingFileError("The file contains less than 2 columns!")

    if reader.line_num < 2:
        raise err.ParsingFileError("The file contains less than 2 lines!")

    for line in reader:
        if len(line) < header_len:
            raise err.ParsingFileError("The file contains lines with missing columns!")

    try:
        df = pd.read_csv(buff, header=0, skipinitialspace=True)
        object_cols = df.columns[df.dtypes == "object"].to_list()

        # Do we still need this if we check csv beforehand?
        if df.empty:
            raise err.FileEmptyError("The uploaded file is empty!")

        if object_cols:
            data.has_object_cols = True

        if df.isna().values.any():
            data.has_nans = True

        data.file = df
        refresh_data(data)
    except ValueError:
        raise err.UploadError("Unable to upload the file!")


def refresh_data(data: data_cls.Data) -> None:
    """Refresh the attributes of the data container.

    Parameters
    ----------
    data : Data
        Data container object.
    """
    data.columns = list(data.file.columns)
    data.columns_counter = clns.Counter(data.columns)
    data.available_columns = data.columns.copy()


def show_data_stats(data: data_cls.Data) -> pd.DataFrame | None:
    """Show the summary of the data statistics.

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
    """Show the data plot.

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
    """Get the columns attached to the layer.

    Parameters
    ----------
    layer_type : "Input" or "Output"
        Type of the layer.
    layer : str
        Name of the layer to which the columns are attached.
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
    """Set the columns for the layer.

    Parameters
    ----------
    layer_type : "Input" or "Output"
        Type of the layer.
    layer : str
        Name of the layer to which the columns will be attached.
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


def layers_are_filled(
    layer_type: Literal["Input", "Output"], data: data_cls.Data, model: model_cls.Model
) -> bool:
    """Check if the specific type layers are filled with the data columns.

    Parameters
    ----------
    layer_type : "Input" or "Output"
        Type of the layers.
    data : Data
        Data container object.
    model : Model
        Model container object.

    Returns
    -------
    bool
        True if the layers are filled with the data columns, False otherwise.
    """
    if layer_type == "Input":
        layers = model.input_layers
        columns_type = data.input_columns
        shape = model.input_shapes
    else:
        layers = model.output_layers
        columns_type = data.output_columns
        shape = model.output_shapes

    for layer in layers:
        if not columns_type.get(layer):
            return False

        if len(columns_type[layer]) < shape[layer]:
            return False

    return True


def split_data(test_size: float, data: data_cls.Data, model: model_cls.Model) -> None:
    """Split the data into training and test sets.

    Parameters
    ----------
    test_size : float
        Test data percent.
    data : Data
        Data container object.
    model : Model
        Model container object.

    Raises
    ------
    InputsUnderfilledError
        When some of the input layers are not filled with the data columns.
    OutputsUnderfilledError
        When some of the output layers are not filled with the data columns.
    """
    if not layers_are_filled("Input", data, model):
        raise err.InputsUnderfilledError(
            "Please, set the data columns for all the input layers!"
        )

    if not layers_are_filled("Output", data, model):
        raise err.OutputsUnderfilledError(
            "Please, set the data columns for all the output layers!"
        )

    try:
        train, test = sk.train_test_split(data.file, test_size=test_size)
    except ValueError as error:
        raise err.IncorrectTestDataPercentage(
            f"Incorrect percent of test data!"
        ) from error

    data.input_train_data = {
        name: train[values].to_numpy() for name, values in data.input_columns.items()
    }
    data.output_train_data = {
        name: train[values].to_numpy() for name, values in data.output_columns.items()
    }
    data.input_test_data = {
        name: test[values].to_numpy() for name, values in data.input_columns.items()
    }
    data.output_test_data = {
        name: test[values].to_numpy() for name, values in data.output_columns.items()
    }
