import streamlit as st
import widgets.upload.upload_file as up_file
import widgets.upload.upload_model as up_model
from data_classes.data import Data
from data_classes.model import Model

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = Data()
    st.session_state.model = Model()

with st.container():
    up_file.create_upload_file_ui(st.session_state.data)
    st.divider()
    up_model.create_upload_model_ui(st.session_state.model)
