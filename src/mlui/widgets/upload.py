import streamlit as st

import mlui.classes.data as data
import mlui.classes.errors as errors
import mlui.classes.model as model


def upload_data_ui(data: data.Data) -> None:
    """Generate the UI for uploading a data file.

    Parameters
    ----------
    data : Data
        Data object.
    """
    st.header("Upload Data")
    st.markdown(
        "Upload a data file from your system using the provided form. You will be able "
        "to view your dataset on the `Data` page. Please note that errors may occur "
        "during the upload if the file is poorly formatted or lacks sufficient rows "
        "and columns. Additionally, be aware that files containing NaN and/or "
        "non-numeric values may result in errors during the training, evaluation, or "
        "prediction processes."
    )

    buff = st.file_uploader("Choose a data file:", "csv", key="file_uploader")

    if buff:
        try:
            data.upload(buff)
            st.toast("File is uploaded!", icon="✅")
        except errors.UploadError as error:
            st.toast(error, icon="❌")


def upload_model_ui(model: model.UploadedModel) -> None:
    """Generate the UI for uploading a model file.

    Parameters
    ----------
    model : UploadedModel
        Model object.
    """
    st.header("Upload Model")
    st.markdown(
        "Upload a model file from your system using the provided form. You will be "
        "able to view your model on the `Model` page. Please note that if the model "
        "was previously compiled, its configuration will not be loaded. Additionally, "
        "be aware that models with more than one dimension in their inputs or outputs, "
        "except for the batch size, are not supported and may cause errors."
    )

    buff = st.file_uploader("Choose a model file:", "h5", key="model_uploader")

    if buff:
        try:
            model.upload(buff)
            st.toast("Model is uploaded!", icon="✅")
        except errors.UploadError as error:
            st.toast(error, icon="❌")
