import data_classes.data as data_cls
import managers.data_manager as dm
import streamlit as st


def show_dataframe_ui(data: data_cls.Data) -> None:
    """Generate UI for displaying a dataframe.

    Parameters
    ----------
    data : Data
        Data container object.
    """
    st.header("Dataframe")
    st.markdown(
        "Here, you can view the uploaded file's dataframe and edit it. "
        "You can sort and resize columns, search through data (click on table, "
        "then `⌘ Cmd + F` or `Ctrl + F`), as well as edit each cell, copy/paste "
        "different parts to/from clipboard."
    )

    data.file = st.data_editor(data.file, use_container_width=True)


def show_data_stats_ui(data: data_cls.Data) -> None:
    """Generate UI for displaying data statistics.

    Parameters
    ----------
    data : Data
        Data container object.
    """
    st.header("Data Statistics")
    st.markdown(
        "Here, you can view the descriptive statistics of the dataset, as well as "
        "the type and number of missing values for each column."
    )

    df = dm.show_data_stats(data)
    st.dataframe(df, use_container_width=True)


def show_data_plot_ui(data: data_cls.Data) -> None:
    """Generate UI for plotting data columns.

    Parameters
    ----------
    data : Data
        Data container object.
    """
    st.header("Data Plot")
    st.markdown(
        "Here, you can plot different columns of the dataset against each other. "
        "A simple line plot is used for two distinct columns, and a histogram "
        "is used for the same ones."
    )

    col_1, col_2 = st.columns(2)

    with col_1:
        x = st.selectbox("Select X-axis:", data.columns)

    with col_2:
        y = st.selectbox("Select Y-axis:", data.columns)

    if x and y:
        fig = dm.show_data_plot(x, y, data)
        st.plotly_chart(fig, use_container_width=True)
