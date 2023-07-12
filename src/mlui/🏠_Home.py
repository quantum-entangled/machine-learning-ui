import streamlit as st

from .data_classes import data as data_cls
from .data_classes import model as model_cls

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

st.set_page_config(page_title="Welcome!", page_icon="🏠")
st.write("# Welcome to Machine Learning UI! 👋")
st.write("Select a page in the sidebar.")
