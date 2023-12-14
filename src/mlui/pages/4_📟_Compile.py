import streamlit as st

import mlui.widgets.compile as widgets
from mlui.decorators.pages import check_model, check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Compile", page_icon="📟")


@set_task
@set_classes
@check_task(["Train", "Evaluate"])
@check_model(["input_configured", "output_configured"])
def compile_page() -> None:
    model = st.session_state.model

    with st.container():
        widgets.set_optimizer_ui(model)
        widgets.set_loss_functions_ui(model)
        widgets.set_metrics_ui(model)
        widgets.compile_model_ui(model)


if __name__ == "__main__":
    compile_page()
