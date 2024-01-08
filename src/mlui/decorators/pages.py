import functools

import streamlit as st

import mlui.types.classes as t


def check_task(page_tasks: t.PageTasks) -> t.DecorType:
    def decorator(func: t.FuncType) -> t.FuncType:
        @functools.wraps(func)
        def wrapper() -> None:
            task = st.session_state.get("task")

            if task not in page_tasks:
                st.info(
                    "The content of this page is not available for the specified task.",
                    icon="💡",
                )
                return

            func()

        return wrapper

    return decorator
