import streamlit as st

from ..data_classes import data as data_cls
from ..data_classes import model as model_cls
from ..managers import model_manager as mm
from ..widgets import model_compilation as mc

st.set_page_config(page_title="Model Compilation", page_icon="📟")

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

data = st.session_state.data
model = st.session_state.model

if not mm.model_exists(model):
    st.info(
        "You will be able to compile the model once you create/upload it.",
        icon="💡",
    )
else:
    with st.container():
        mc.set_optimizer_ui(model)
        mc.set_loss_functions_ui(model)
        mc.set_metrics_ui(model)
        mc.compile_model_ui(model)
