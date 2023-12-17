import streamlit as st

import mlui.decorators as decorators
import mlui.widgets.evaluate as widgets

st.set_page_config(page_title="Evaluate", page_icon="✔️")


@decorators.session.set_state
@decorators.pages.check_task(["Evaluate"])
def evaluate_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    if not model.compiled:
        st.info(
            "The content of this page will be available once the model is compiled.",
            icon="💡",
        )
        return

    with st.container():
        widgets.evaluate_model_ui(data, model)


if __name__ == "__main__":
    evaluate_page()
