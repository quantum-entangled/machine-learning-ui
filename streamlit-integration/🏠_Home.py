import streamlit as st
from data_classes.data import Data
from data_classes.model import Model

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = Data()
    st.session_state.model = Model()

st.set_page_config(page_title="Welcome!", page_icon="👋")
st.write("# Welcome to Machine Learning UI! 👋")
st.write("Select a page in the sidebar.")
