import streamlit as st

import mlui.widgets.upload as up
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Upload", page_icon="⬆️")


@set_task
@set_classes
def upload_page() -> None:
    with st.container():
        up.upload_file_ui(st.session_state.data)

        if st.session_state.task != "Training":
            up.upload_model_ui(st.session_state.model)


if __name__ == "__main__":
    upload_page()
