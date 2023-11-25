import functools

import streamlit as st


# TODO: Add docstrings and type hints
def check_task(page_tasks: list[str]):
    def decorator(func):
        @functools.wraps(func)
        def inner():
            if st.session_state.task not in page_tasks:
                st.info(
                    "The content of this page is not available for the specified task.",
                    icon="💡",
                )
            else:
                func()

        return inner

    return decorator
