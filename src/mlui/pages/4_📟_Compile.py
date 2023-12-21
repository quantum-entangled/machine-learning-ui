import streamlit as st

import mlui.decorators as decorators
import mlui.widgets.compile as widgets

st.set_page_config(page_title="Compile", page_icon="📟")


@decorators.session.set_state
@decorators.pages.check_task(["Train", "Evaluate"])
def compile_page() -> None:
    model = st.session_state.model

    if not model.input_configured or not model.output_configured:
        st.info(
            "The content of this page will be available "
            "once the model's layers are configured.",
            icon="💡",
        )
        return

    with st.container():
        widgets.set_optimizer_ui(model)
        widgets.set_loss_functions_ui(model)
        widgets.set_metrics_ui(model)
        widgets.compile_model_ui(model)


if __name__ == "__main__":
    compile_page()
