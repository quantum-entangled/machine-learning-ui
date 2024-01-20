import streamlit as st

import mlui.decorators as decorators
import mlui.widgets.create as widgets

st.set_page_config(page_title="Create", page_icon="🛠️")


@decorators.session.set_state
@decorators.pages.check_task(["Train"])
def create_page() -> None:
    model = st.session_state.model
    model_type = st.session_state.model_type

    if model_type != "Created":
        st.info(
            "The content of this page is not available for the specified model type.",
            icon="💡",
        )
        return

    with st.container():
        widgets.set_name_ui(model)
        widgets.set_layers_ui(model)
        widgets.set_outputs_ui(model)
        widgets.create_model_ui(model)


if __name__ == "__main__":
    create_page()
