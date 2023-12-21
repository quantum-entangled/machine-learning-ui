import streamlit as st

import mlui.classes.model as model
import mlui.decorators as decorators

st.set_page_config(page_title="Welcome!", page_icon="🏠")


@decorators.session.set_state
def home_page() -> None:
    st.write("# Welcome! 👋")
    st.markdown(
        "💠 On this page, you can select the task you want to perform as well as "
        "the type of model you will use. These selections will influence the content "
        "of the pages you see in the sidebar on the left. Some pages might not be "
        "available for the specified task or model type, or they might require "
        "specific actions to be loaded. In that case, you will see an info box "
        "containing the requirements/constraints for the page."
    )
    st.markdown(
        "💠 `Data` and `Model` pages are always available. They contain the "
        "essential information about two main instances for the whole app session."
    )
    st.markdown(
        "💠 You will be guided by the instructions for each respective section. In "
        "most cases, when you click a button responsible for the main action in the "
        "section, a message of either success or error will be displayed in the "
        "bottom right corner of the screen, providing you with essential information "
        "about the executed process."
    )
    st.markdown(
        "💠 Each time you press the `R` button on your keyboard, the page's "
        "content, including all forms, messages, and displayed elements, will be "
        "refreshed. However, if you click the `Reload` button in your browser panel, "
        "the entire app session will be reloaded, meaning you will lose all "
        "information and content available up to that moment."
    )

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
