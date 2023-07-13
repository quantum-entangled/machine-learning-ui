import streamlit as st

import mlui.data_classes.data as data_cls
import mlui.data_classes.model as model_cls
import mlui.managers.model_manager as mm
import mlui.widgets.training as tr

st.set_page_config(page_title="Training", page_icon="🏋️‍♂️")

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

data = st.session_state.data
model = st.session_state.model

if not mm.model_exists(model) or not model.compiled:
    st.info(
        "You will be able to train the model once you create/upload and compile it.",
        icon="💡",
    )
else:
    with st.container():
        tr.set_callbacks_ui(model)
        tr.fit_model_ui(data, model)
        tr.show_history_plot_ui(model)
