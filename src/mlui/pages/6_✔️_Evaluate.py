import streamlit as st

import mlui.widgets.model_evaluation as me
import mlui.widgets.training as tr
from mlui.decorators.pages import check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Evaluate", page_icon="✔️")


@set_task
@set_classes
@check_task(["Evaluation"])
def evaluate_page() -> None:
    # TODO: Check for existence of the model; check that the model is compiled
    with st.container():
        tr.set_callbacks_ui(st.session_state.model)
        me.evaluate_model_ui(st.session_state.data, st.session_state.model)


if __name__ == "__main__":
    evaluate_page()
