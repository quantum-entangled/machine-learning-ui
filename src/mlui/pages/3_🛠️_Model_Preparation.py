import streamlit as st

from ..data_classes import data as data_cls
from ..data_classes import model as model_cls
from ..widgets import model_preparation as mp

st.set_page_config(page_title="Model Preparation", page_icon="🛠️")

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

data = st.session_state.data
model = st.session_state.model

with st.container():
    mp.create_model_ui(model)
    mp.add_layers_ui(model)
    mp.set_outputs_ui(model)
    mp.examine_model(model)
