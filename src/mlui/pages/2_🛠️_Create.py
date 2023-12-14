import streamlit as st

import mlui.widgets.create as widgets
from mlui.decorators.pages import check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Create", page_icon="🛠️")


@set_task
@set_classes
@check_task(["Train"])
def create_page() -> None:
    model = st.session_state.model

    with st.container():
        widgets.set_name_ui(model)
        widgets.set_layers_ui(model)
        widgets.set_outputs_ui(model)
        widgets.create_model_ui(model)


if __name__ == "__main__":
    create_page()
