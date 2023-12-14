import streamlit as st

import mlui.widgets.predict as widgets
from mlui.decorators.pages import check_model, check_task
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Predict", page_icon="🔮")


@set_task
@set_classes
@check_task(["Predict"])
@check_model(["input_configured"])
def predict_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    with st.container():
        widgets.make_predictions_ui(data, model)


if __name__ == "__main__":
    predict_page()
