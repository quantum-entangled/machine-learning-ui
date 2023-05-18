import data_classes.data as data_cls
import data_classes.model as model_cls
import streamlit as st
import widgets.data_preparation as dp

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = data_cls.Data()
    st.session_state.model = model_cls.Model()

data = st.session_state.data
model = st.session_state.model

with st.container():
    dp.show_dataframe_ui(data)
    dp.show_data_stats_ui(data)
    dp.show_data_plot_ui(data)
