import streamlit as st

from mlui.classes.model import CreatedModel, UploadedModel
from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Welcome!", page_icon="🏠")


@set_task
@set_classes
def home_page() -> None:
    st.write("# Welcome to Machine Learning UI! 👋")

    tasks = ("Train", "Evaluate", "Predict")
    current_task = tasks.index(st.session_state.task)

    with st.form("set_task_form", border=False):
        task = st.selectbox("Select a task:", tasks, current_task)
        set_task_btn = st.form_submit_button("Set Task")

    if set_task_btn:
        st.session_state.task = task

        st.session_state.data.update_state()

        if st.session_state.task == "Train":
            st.session_state.model = CreatedModel()
        else:
            st.session_state.model = UploadedModel()

        st.toast("Task is set!", icon="✅")


if __name__ == "__main__":
    home_page()
