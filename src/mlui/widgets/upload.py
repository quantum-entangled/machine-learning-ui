import streamlit as st

from mlui.classes.data import Data
from mlui.classes.errors import UploadError
from mlui.classes.model import UploadedModel


def upload_file_ui(data: Data) -> None:
    """Generate the UI for uploading a file.

    Parameters
    ----------
    data : Data
        Data object.
    """
    st.header("Upload File")
    st.markdown("Upload a file from your system by clicking the `Browse files` button.")

    buff = st.file_uploader("Choose a data file:", "csv", key="file_uploader")

    if buff:
        try:
            data.upload(buff)
            st.toast("File is uploaded!", icon="✅")
        except UploadError as error:
            st.toast(error, icon="❌")


def upload_model_ui(model: UploadedModel) -> None:
    """Generate the UI for uploading a model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Upload Model")
    st.markdown(
        "Upload a model from your system by clicking the `Browse files` button."
    )

    buff = st.file_uploader("Choose a model file:", "h5", key="model_uploader")

    if buff:
        try:
            model.upload(buff)
            st.toast("Model is uploaded!", icon="✅")
        except UploadError as error:
            st.toast(error, icon="❌")
