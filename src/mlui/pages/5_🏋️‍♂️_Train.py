import streamlit as st

import mlui.decorators as decorators
import mlui.widgets.train as widgets

st.set_page_config(page_title="Train", page_icon="🏋️‍♂️")


@decorators.session.set_state
@decorators.pages.check_task(["Train"])
def train_page() -> None:
    data = st.session_state.data
    model = st.session_state.model

    if not model.compiled:
        st.info(
            "The content of this page will be available once the model is compiled.",
            icon="💡",
        )
        return

    with st.container():
        widgets.fit_model_ui(data, model)
        widgets.plot_history_ui(model)


if __name__ == "__main__":
    train_page()
