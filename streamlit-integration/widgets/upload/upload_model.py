import tempfile

import streamlit as st
from data_classes.model import Model

import managers.model_manager as mm


def create_upload_model_ui(model: Model) -> None:
    st.header("Upload Model")
    uploaded_model = st.file_uploader(
        "Choose a model file:", "h5", key="model_uploader"
    )

    if uploaded_model:
        try:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(uploaded_model.getbuffer())
                mm.upload_model(tmp.name, model)

            st.success("Model is uploaded and ready to be processed!", icon="✅")
        except mm.UploadError as error:
            st.exception(error)

    if mm.model_exists(model):
        st.info(
            "View the existing model on Model Preparation page or upload a new one.",
            icon="💡",
        )
