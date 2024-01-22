import streamlit as st

import mlui.decorators as decorators
import mlui.widgets.configure as widgets

st.set_page_config(page_title="Configure", page_icon="⚙️")


@decorators.session.set_state
def configure_page() -> None:
    """Generate a Streamlit app page for configuring the model."""
    data = st.session_state.data
    model = st.session_state.model

    if data.empty or not model.built:
        st.info(
            "The content of this page will be available once the model "
            "is uploaded/created and the data file is uploaded.",
            icon="💡",
        )
        return

    with st.container():
        widgets.set_features_ui(data, model)
        widgets.set_callbacks_ui(model)


if __name__ == "__main__":
    configure_page()
