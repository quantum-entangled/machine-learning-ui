import streamlit as st

import mlui.widgets.model_compilation as mc
from mlui.decorators.pages import check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Compile", page_icon="📟")


@set_task
@set_classes
@check_task(["Training", "Evaluation"])
def compile_page() -> None:
    # TODO: Check for existence of the model
    with st.container():
        mc.set_optimizer_ui(st.session_state.model)
        mc.set_loss_functions_ui(st.session_state.model)
        mc.set_metrics_ui(st.session_state.model)
        mc.compile_model_ui(st.session_state.model)


if __name__ == "__main__":
    compile_page()
