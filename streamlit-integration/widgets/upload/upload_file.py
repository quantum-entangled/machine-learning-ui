import streamlit as st
from data_classes.data import Data

import managers.data_manager as dm


def create_upload_file_ui(data: Data) -> None:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose a data file:", "csv", key="file_uploader")

    if uploaded_file:
        try:
            dm.upload_file(uploaded_file, data)
            st.success("File is uploaded and ready to be processed!", icon="✅")
        except dm.UploadError as error:
            st.exception(error)

    if dm.file_exists(data):
        st.info(
            "View the existing file on Data Preparation page or upload a new one.",
            icon="💡",
        )
