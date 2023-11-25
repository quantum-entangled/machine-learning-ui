import streamlit as st

import mlui.widgets.data_preparation as dp
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Data", page_icon="📝")


@set_task
@set_classes
def data_page() -> None:
    with st.container():
        dp.show_dataframe_ui(st.session_state.data)
        dp.show_data_stats_ui(st.session_state.data)
        dp.show_data_plot_ui(st.session_state.data)


if __name__ == "__main__":
    data_page()
