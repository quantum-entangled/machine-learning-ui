import streamlit as st

import mlui.widgets.upload as widgets
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Upload", page_icon="⬆️")


@set_task
@set_classes
def upload_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    with st.container():
        widgets.upload_file_ui(data)

        if st.session_state.task != "Train":
            widgets.upload_model_ui(model)


if __name__ == "__main__":
    upload_page()
