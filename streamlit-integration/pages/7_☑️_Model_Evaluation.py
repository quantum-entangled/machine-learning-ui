import data_classes.data as data_cls
import data_classes.model as model_cls
import streamlit as st
import widgets.model_evaluation as me
import widgets.training as tr

import managers.model_manager as mm

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
