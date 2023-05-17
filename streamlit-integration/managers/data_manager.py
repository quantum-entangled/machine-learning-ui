import io

import pandas as pd
from data_classes.data import Data
from data_classes.model import Model

from managers.errors import UploadError


def file_exists(data: Data) -> bool:
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


def upload_file(buff: io.BytesIO | None, data: Data) -> None:
    """Read file to pandas format.

    Parameters
    ----------
    buff : BytesIO | None
        Buffer object to upload.
    data : Data
        Data container object.
    """
    if not buff:
        return

    try:
        data.file = pd.read_csv(buff, header=0, skipinitialspace=True)
    except ValueError as error:
        raise UploadError(f"Unable to upload the file!") from error


def refresh_data(data: Data, model: Model) -> None:
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
