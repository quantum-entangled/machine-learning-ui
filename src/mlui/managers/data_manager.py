import collections as clns
import io
import re
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


def is_file_correct(csv: str) -> bool:
    """Checking if file structure is correct.

    Parameters
    ----------
    csv : str
        CSV file as a string.

    Returns
    -------
    bool
        False if file contains :
        - not numbers in columns
        - multi-indexes (different number of columns in rows)
        - less than 2 columns
        - less than 2 lines
        - incorrect indentation or separators

        True otherwise.
    """
    rows = csv.split("\n")
    if len(rows) < 2:
        return False

    columns = rows[0].split(",")
    len_columns = len(columns)
    if len_columns < 2:
        return False

    regex_str = r"^(((((\d+\.\d+)|(\d*)),){%d}((\d+\.\d+)|(\d*)))|)$" % (len_columns - 1)
    regex = re.compile(regex_str)

    for row in rows[1:]:
        if not regex.match(row):
            return False
    return True


def upload_file(
    buff: io.BytesIO | None, data: data_cls.Data, model: model_cls.Model
) -> None:
    """Read a file to the pandas format.

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
    IncorrectFileStructure
        When the file structure is incorrect.
    FileEmptyError
        When trying to upload an empty file.
    """
    if not buff:
        return

    try:
        csv = buff.read().decode("utf-8")

        if is_file_correct(csv):
            data.file = pd.read_csv(
                io.BytesIO(bytes(csv, "utf-8")), header=0, skipinitialspace=True
            )
            refresh_data(data)
        else:
            raise err.IncorrectFileStructure(f"The file has an incorrect structure!")

        if not file_exists(data):
            raise err.FileEmptyError(f"The uploaded file is empty!")
    except ValueError as error:
        raise err.UploadError(f"Unable to upload the file!") from error


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
