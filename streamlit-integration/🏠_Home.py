import data_classes.data as data_cls
import data_classes.model as model_cls
import streamlit as st

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

st.set_page_config(page_title="Welcome!", page_icon="👋")
st.write("# Welcome to Machine Learning UI! 👋")
st.write("Select a page in the sidebar.")
