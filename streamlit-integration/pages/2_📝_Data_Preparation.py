import streamlit as st
from data_classes.data import Data
from data_classes.model import Model
from widgets.data_preparation import data_preparation as dp

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = Data()
    st.session_state.model = Model()

data = st.session_state.data
model = st.session_state.model

with st.container():
    dp.show_dataframe_ui(data)
    dp.show_data_stats_ui(data)
    dp.show_data_plot_ui(data)
