import streamlit as st

from ..data_classes import data as data_cls
from ..data_classes import model as model_cls
from ..managers import data_manager as dm
from ..managers import model_manager as mm
from ..widgets import predictions as pr
from ..widgets import training as tr

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
