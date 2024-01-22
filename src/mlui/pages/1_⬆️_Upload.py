import streamlit as st

import mlui.decorators as decorators
import mlui.widgets.upload as widgets

st.set_page_config(page_title="Upload", page_icon="⬆️")


@decorators.session.set_state
def upload_page() -> None:
    """Generate a Streamlit app page for uploading the data and the model."""
    data = st.session_state.data
    model = st.session_state.model
    model_type = st.session_state.model_type

    with st.container():
        widgets.upload_data_ui(data)

        if model_type == "Uploaded":
            widgets.upload_model_ui(model)


if __name__ == "__main__":
    upload_page()
