import streamlit as st
from streamlit_extras.capture import stdout
from streamlit_extras.chart_container import chart_container

from mlui.classes.data import Data
from mlui.classes.errors import ModelError, PlotError
from mlui.classes.model import CreatedModel


def fit_model_ui(data: Data, model: CreatedModel) -> None:
    """Generate the UI for fitting the model.

    Parameters
    ----------
    data : Data
        Data object.
    model : Model
        Model object.
    """
    st.header("Fit Model")

    batch_size = st.number_input(
        "Batch size:", min_value=1, max_value=1024, value=32, step=1
    )
    num_epochs = st.number_input(
        "Number of epochs:", min_value=1, max_value=1000, value=30, step=1
    )
    val_split = st.number_input(
        "Validation split:", min_value=0.01, max_value=1.0, value=0.15, step=0.01
    )
    fit_model_btn = st.button("Fit Model")

    if fit_model_btn:
        with st.status("Training Logs"):
            with stdout(st.empty().code):
                try:
                    df = data.dataframe

                    model.fit(df, int(batch_size), int(num_epochs), float(val_split))
                    st.toast("Training is completed!", icon="✅")
                except ModelError as error:
                    st.toast(error, icon="❌")


def plot_history_ui(model: CreatedModel) -> None:
    """Generate the UI for plotting the training history.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Plot History")

    history = model.history
    options = history.columns.drop("epoch") if not history.empty else list()

    with st.form("plot_history_form", border=False):
        y = st.multiselect("Select Y-axis log(s):", options)
        points = st.toggle("Point Markers")
        plot_history_btn = st.form_submit_button("Plot History")

    if plot_history_btn:
        with chart_container(history, export_formats=["CSV"]):
            try:
                chart = model.plot_history(y, points)

                st.altair_chart(chart, use_container_width=True)
            except PlotError as error:
                st.toast(error, icon="❌")
