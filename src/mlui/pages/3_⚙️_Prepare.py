import streamlit as st

import mlui.widgets.training_preparation as tp
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Prepare", page_icon="⚙️")


@set_task
@set_classes
def prepare_page() -> None:
    # TODO: Check for existence of the data and the model
    with st.container():
        tp.set_columns_ui(st.session_state.data, st.session_state.model)
        tp.split_data_ui(st.session_state.data, st.session_state.model)


if __name__ == "__main__":
    prepare_page()
