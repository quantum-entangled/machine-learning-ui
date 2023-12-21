import functools

import streamlit as st

import mlui.types.classes as t


# TODO: Add type hints
def check_task(page_tasks: t.PageTasks):
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
