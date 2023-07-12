import streamlit as st

from ..data_classes import data as data_cls
from ..data_classes import model as model_cls
from ..widgets import upload as up

st.set_page_config(page_title="Uploading", page_icon="⬆️")

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

data = st.session_state.data
model = st.session_state.model

with st.container():
    up.upload_file_ui(data, model)
    up.upload_model_ui(model)
