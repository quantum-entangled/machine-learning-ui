import streamlit as st

import mlui.widgets.data as widgets
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Data", page_icon="📝")


@set_task
@set_classes
def data_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    with st.container():
        widgets.data_info_ui(data)
        widgets.dataframe_ui(data)
        widgets.statistics_ui(data)
        widgets.plot_columns_ui(data)
        widgets.reset_data_ui(data, model)


if __name__ == "__main__":
    data_page()
