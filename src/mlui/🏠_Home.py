import streamlit as st

from mlui.decorators.session import set_classes, set_task

st.set_page_config(page_title="Welcome!", page_icon="🏠")


@set_task
@set_classes
def home_page() -> None:
    st.write("# Welcome to Machine Learning UI! 👋")

    # BUG: New task doesn't show up in Selectbox when changing it multiple times
    tasks = ("Training", "Evaluation", "Predictions")
    current_task = tasks.index(st.session_state.task)
    st.session_state.task = st.selectbox("Select a task:", tasks, current_task)


if __name__ == "__main__":
    home_page()
