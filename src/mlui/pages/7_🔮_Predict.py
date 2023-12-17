import streamlit as st

import mlui.decorators as decorators
import mlui.widgets.predict as widgets

st.set_page_config(page_title="Predict", page_icon="🔮")


@decorators.session.set_state
@decorators.pages.check_task(["Predict"])
def predict_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    if not model.input_configured:
        st.info(
            "The content of this page will be available "
            "once the model's input layers are configured.",
            icon="💡",
        )
        return

    with st.container():
        widgets.make_predictions_ui(data, model)


if __name__ == "__main__":
    predict_page()
