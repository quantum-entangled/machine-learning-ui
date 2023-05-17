import streamlit as st
from data_classes.data import Data
from data_classes.model import Model
from widgets.upload import upload as up

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = Data()
    st.session_state.model = Model()

with st.container():
    up.create_upload_file_ui(st.session_state.data)
    st.divider()
    up.create_upload_model_ui(st.session_state.model)
