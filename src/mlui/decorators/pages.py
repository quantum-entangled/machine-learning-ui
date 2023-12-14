import functools

import streamlit as st


# TODO: Add docstrings and type hints
def check_task(page_tasks: list[str]):
    def decorator(func):
        @functools.wraps(func)
        def inner():
            task = st.session_state.get("task")

            if task not in page_tasks:
                st.info(
                    "The content of this page is not available for the specified task.",
                    icon="💡",
                )
                return

            func()

        return inner

    return decorator


def check_model(states: list[str]):
    def decorator(func):
        @functools.wraps(func)
        def inner():
            model = st.session_state["model"]

            if "built" in states and not model.built:
                st.info(
                    "The content of this page will be available "
                    "once the model is uploaded or created.",
                    icon="💡",
                )
                return

            if "output_configured" in states and not model.output_configured:
                st.info(
                    "The content of this page will be available "
                    "once the model's output layers are configured.",
                    icon="💡",
                )
                return

            if "input_configured" in states and not model.input_configured:
                st.info(
                    "The content of this page will be available "
                    "once the model's input layers are configured.",
                    icon="💡",
                )
                return

            if "compiled" in states and not model.compiled:
                st.info(
                    "The content of this page will be available "
                    "once the model is compiled.",
                    icon="💡",
                )
                return

            func()

        return inner

    return decorator
