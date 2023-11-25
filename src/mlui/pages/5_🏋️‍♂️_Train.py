import streamlit as st

import mlui.widgets.training as tr
from mlui.decorators.pages import check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Train", page_icon="🏋️‍♂️")


@set_task
@set_classes
@check_task(["Training"])
def train_page() -> None:
    # TODO: Check for existence of the model; check that the model is compiled
    with st.container():
        tr.set_callbacks_ui(st.session_state.model)
        tr.fit_model_ui(st.session_state.data, st.session_state.model)
        tr.show_history_plot_ui(st.session_state.model)


if __name__ == "__main__":
    train_page()
