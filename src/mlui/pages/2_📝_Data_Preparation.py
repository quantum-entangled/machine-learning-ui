import streamlit as st

from ..data_classes import data as data_cls
from ..data_classes import model as model_cls
from ..widgets import data_preparation as dp

st.set_page_config(page_title="Data Preparation", page_icon="📝")

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

data = st.session_state.data
model = st.session_state.model

with st.container():
    dp.show_dataframe_ui(data)
    dp.show_data_stats_ui(data)
    dp.show_data_plot_ui(data)
