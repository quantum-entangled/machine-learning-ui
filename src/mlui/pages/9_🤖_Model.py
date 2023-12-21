import streamlit as st

import mlui.decorators as decorators
import mlui.widgets.model as widgets

st.set_page_config(page_title="Model", page_icon="🤖")


@decorators.session.set_state
def model_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    with st.container():
        widgets.model_info_ui(model)
        widgets.summary_ui(model)
        widgets.graph_ui(model)
        widgets.download_model_ui(model)
        widgets.reset_model_ui(data, model)


if __name__ == "__main__":
    model_page()
