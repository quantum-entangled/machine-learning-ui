import data_classes.data as data_cls
import data_classes.model as model_cls
import managers.data_manager as dm
import managers.model_manager as mm
import streamlit as st
import widgets.training_preparation as tp

st.set_page_config(page_title="Training Preparation", page_icon="⚙️")

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

data = st.session_state.data
model = st.session_state.model

if not dm.file_exists(data) or not mm.model_exists(model):
    st.info(
        "You will be able to prepare for the training process once you upload the data "
        "file and create/upload the model.",
        icon="💡",
    )
else:
    with st.container():
        tp.set_columns_ui(data, model)
        tp.split_data_ui(data, model)
