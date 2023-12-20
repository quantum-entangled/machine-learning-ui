import streamlit as st

import mlui.classes.model as model
import mlui.decorators as decorators

st.set_page_config(page_title="Welcome!", page_icon="🏠")


@decorators.session.set_state
def home_page() -> None:
    st.write("# Welcome! 👋")

    tasks = ("Train", "Evaluate", "Predict")
    default = tasks.index(st.session_state.task)
    task = st.selectbox("Select task:", tasks, default)

    if task == "Train":
        types = ("Created", "Uploaded")
        default = types.index(st.session_state.model_type)
    else:
        types = ("Uploaded",)
        default = 0

    model_type = st.selectbox("Select model type:", types, default)

    def set_state() -> None:
        """Supporting function for the accurate representation of widgets."""
        st.session_state.task = task
        st.session_state.model_type = model_type
        st.session_state.model = (
            model.CreatedModel() if model_type == "Created" else model.UploadedModel()
        )

        st.session_state.data.update_state()
        st.toast("State is set!", icon="✅")

    st.button("Set State", on_click=set_state)


if __name__ == "__main__":
    home_page()
