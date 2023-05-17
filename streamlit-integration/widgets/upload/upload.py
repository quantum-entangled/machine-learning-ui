import streamlit as st
from data_classes.data import Data
from data_classes.model import Model

import managers.data_manager as dm
import managers.model_manager as mm


def upload_file_ui(data: Data, model: Model) -> None:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose a data file:", "csv", key="file_uploader")

    if uploaded_file:
        try:
            dm.upload_file(uploaded_file, data, model)
            st.success("File is uploaded and ready to be processed!", icon="✅")
        except dm.UploadError as error:
            st.exception(error)

    if dm.file_exists(data):
        st.info(
            "View the existing file on Data Preparation page or upload a new one.",
            icon="💡",
        )


def upload_model_ui(model: Model) -> None:
    st.header("Upload Model")
    uploaded_model = st.file_uploader(
        "Choose a model file:", "h5", key="model_uploader"
    )

    if uploaded_model:
        try:
            mm.upload_model(uploaded_model, model)
            st.success("Model is uploaded and ready to be processed!", icon="✅")
        except mm.UploadError as error:
            st.exception(error)

    if mm.model_exists(model):
        st.info(
            "View the existing model on Model Preparation page or upload a new one.",
            icon="💡",
        )
