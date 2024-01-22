import functools

import streamlit as st

import mlui.types.classes as t


def check_task(page_tasks: t.PageTasks) -> t.DecorType:
    """
    Decorator to check if the provided tasks are allowed for the page.

    Parameters
    ----------
    page_tasks : list of str
        Tasks to check.
    func : Callable
        Function to wrap and execute.

    Returns
    -------
    Callable
        Wrapped function.
    """

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
