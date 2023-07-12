import streamlit as st

from ..data_classes import data as data_cls
from ..data_classes import model as model_cls
from ..managers import model_manager as mm
from ..widgets import model_evaluation as me
from ..widgets import training as tr

st.set_page_config(page_title="Model Evaluation", page_icon="☑️")

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

data = st.session_state.data
model = st.session_state.model

if not mm.model_exists(model) or not model.compiled:
    st.info(
        "You will be able to evaluate the model once you create/upload and compile it.",
        icon="💡",
    )
else:
    with st.container():
        tr.set_callbacks_ui(model)
        me.evaluate_model_ui(data, model)
