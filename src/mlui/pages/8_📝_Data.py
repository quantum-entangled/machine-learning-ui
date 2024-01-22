import streamlit as st

import mlui.decorators as decorators
import mlui.widgets.data as widgets

st.set_page_config(page_title="Data", page_icon="📝")


@decorators.session.set_state
def data_page() -> None:
    """Generate a Streamlit app page for examining the data."""
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
