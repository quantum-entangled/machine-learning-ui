import streamlit as st

import mlui.classes.data as data
import mlui.classes.errors as errors
import mlui.classes.model as model


def data_info_ui(data: data.Data) -> None:
    """Generate the UI for displaying data information.

    Parameters
    ----------
    data : Data
        Data object.
    """
    st.header("Data Info")

    empty = data.empty
    has_nans = data.has_nans
    has_nonnumeric_dtypes = data.has_nonnumeric_dtypes

    if empty:
        st.info("The data file is not uploaded.", icon="💡")
        return

    if has_nans:
        st.info("The DataFrame contains `NaN` values.", icon="💡")
    else:
        st.success("The DataFrame doesn't contain `NaN` values.", icon="✅")

    if has_nonnumeric_dtypes:
        st.info("The DataFrame contains non-numeric values.", icon="💡")
    else:
        st.success("The DataFrame doesn't contain non-numeric values.", icon="✅")


def dataframe_ui(data: data.Data) -> None:
    """Generate the UI for displaying a dataframe.

    Parameters
    ----------
    data : Data
        Data object.
    """
    st.header("Dataframe")
    st.markdown(
        "Here, you can view the uploaded file's dataframe and edit it. "
        "You can sort and resize columns, search through data (click on table, "
        "then `⌘ Cmd + F` or `Ctrl + F`), as well as edit each cell, copy/paste "
        "different parts to/from clipboard."
    )

    st.dataframe(data.dataframe, use_container_width=True)


def statistics_ui(data: data.Data) -> None:
    """Generate the UI for displaying data statistics.

    Parameters
    ----------
    data : Data
        Data object.
    """
    st.header("Statistics")
    st.markdown(
        "Here, you can view the descriptive statistics of the dataset, as well as "
        "the type and number of missing values for each column."
    )

    stats = data.get_stats()
    st.dataframe(stats, use_container_width=True)


def plot_columns_ui(data: data.Data) -> None:
    """Generate the UI for plotting data columns.

    Parameters
    ----------
    data : Data
        Data object.
    """
    st.header("Plot Columns")
    st.markdown(
        "Here, you can plot different columns of the dataset against each other. "
        "A simple line plot is used for two distinct columns, and a histogram "
        "is used for the same ones."
    )

    with st.form("plot_columns_form", border=False):
        x = st.selectbox("Select X-axis column:", data.columns)
        y = st.selectbox("Select Y-axis column:", data.columns)
        points = st.toggle("Point Markers")
        plot_columns_btn = st.form_submit_button("Plot Columns")

    if plot_columns_btn:
        try:
            chart = data.plot_columns(x, y, points)

            st.altair_chart(chart, use_container_width=True)
        except errors.PlotError as error:
            st.toast(error, icon="❌")


def reset_data_ui(data: data.Data, model: model.Model) -> None:
    """Generate the UI for resetting the data.

    Parameters
    ----------
    data : Data
        Data object.
    model : Model
        Model object.
    """
    st.header("Reset Data")

    reset_data_btn = st.button("Reset Data")

    if reset_data_btn:
        data.reset_state()
        model.update_state()
        st.rerun()
