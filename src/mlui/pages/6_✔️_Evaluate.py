import streamlit as st

import mlui.widgets.evaluate as widgets
from mlui.decorators.pages import check_model, check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Evaluate", page_icon="✔️")


@set_task
@set_classes
@check_task(["Evaluate"])
@check_model(["compiled"])
def evaluate_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    with st.container():
        widgets.evaluate_model_ui(data, model)


if __name__ == "__main__":
    evaluate_page()
