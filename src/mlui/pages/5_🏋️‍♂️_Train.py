import streamlit as st

import mlui.widgets.train as widgets
from mlui.decorators.pages import check_model, check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Train", page_icon="🏋️‍♂️")


@set_task
@set_classes
@check_task(["Train"])
@check_model(["compiled"])
def train_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    with st.container():
        widgets.fit_model_ui(data, model)
        widgets.plot_history_ui(model)


if __name__ == "__main__":
    train_page()
