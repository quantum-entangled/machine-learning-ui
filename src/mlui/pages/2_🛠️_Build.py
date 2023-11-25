import streamlit as st

import mlui.widgets.model_preparation as mp
from mlui.decorators.pages import check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Build", page_icon="🛠️")


@set_task
@set_classes
@check_task(["Training"])
def build_page() -> None:
    with st.container():
        mp.create_model_ui(st.session_state.model)
        mp.add_layers_ui(st.session_state.model)
        mp.set_outputs_ui(st.session_state.model)
        mp.examine_model(st.session_state.model)


if __name__ == "__main__":
    build_page()
