import streamlit as st

import mlui.widgets.predictions as pr
import mlui.widgets.training as tr
from mlui.decorators.pages import check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Predict", page_icon="🔮")


@set_task
@set_classes
@check_task(["Predictions"])
def predict_page() -> None:
    # TODO: Check for existence of the data and the model
    with st.container():
        tr.set_callbacks_ui(st.session_state.model)
        pr.make_predictions_ui(st.session_state.data, st.session_state.model)


if __name__ == "__main__":
    predict_page()
