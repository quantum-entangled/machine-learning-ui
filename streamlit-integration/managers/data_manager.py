import io

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
    buff : File-like object | None
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
        refresh_data(data, model)
    except ValueError as error:
        raise err.UploadError(f"Unable to upload a file!") from error


def refresh_data(data: data_cls.Data, model: model_cls.Model) -> None:
    """Refresh attributes of data container.

    Parameters
    ----------
    data : Data
        Data container object.
    model : Model
        Model container object.
    """
    data.columns = list(data.file.columns)
    data.input_columns = {name: list() for name in model.input_layers}
    data.output_columns = {name: list() for name in model.output_layers}
    data.columns_per_layer = {
        name: 0 for name in model.input_layers | model.output_layers
    }
    data.input_train_data = dict()
    data.output_train_data = dict()
    data.input_test_data = dict()
    data.output_test_data = dict()


def show_data_stats(data: data_cls.Data) -> pd.DataFrame | None:
    """Show summary of data statistics.

    Parameters
    ----------
    data : Data
        Data container object.

    Returns
    -------
    DataFrame | None
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
