import streamlit as st

import mlui.widgets.configure as widgets
from mlui.decorators.pages import check_model
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Configure", page_icon="⚙️")


@set_task
@set_classes
@check_model(["built"])
def prepare_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    with st.container():
        widgets.set_features_ui(data, model)
        widgets.set_callbacks_ui(model)


if __name__ == "__main__":
    prepare_page()
