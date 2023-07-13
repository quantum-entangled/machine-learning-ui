import streamlit as st

import mlui.data_classes.data as data_cls
import mlui.data_classes.model as model_cls
import mlui.managers.data_manager as dm
import mlui.managers.model_manager as mm
import mlui.widgets.predictions as pr
import mlui.widgets.training as tr

st.set_page_config(page_title="Predictions", page_icon="🔮")

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

data = st.session_state.data
model = st.session_state.model

if not dm.file_exists(data) or not mm.model_exists(model):
    st.info(
        "You will be able to make predictions once you upload the data file, as well "
        "as create/upload and compile the model.",
        icon="💡",
    )
else:
    with st.container():
        tr.set_callbacks_ui(model)
        pr.make_predictions_ui(data, model)
