import streamlit as st

import mlui.data_classes.data as data_cls
import mlui.data_classes.model as model_cls
import mlui.widgets.model_preparation as mp

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
