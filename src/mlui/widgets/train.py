import streamlit as st
import streamlit_extras.capture as capture
import streamlit_extras.chart_container as container

import mlui.classes.data as data
import mlui.classes.errors as errors
import mlui.classes.model as model


def fit_model_ui(data: data.Data, model: model.CreatedModel) -> None:
    """Generate the UI for fitting the model.

    Parameters
    ----------
    data : Data
        Data object.
    model : Model
        Model object.
    """
    st.header("Fit Model")
    st.markdown(
        "Train the model by specifying the required hyperparameters. Once the "
        "`Fit Model` button is clicked, the training process will start, and logs "
        "will be displayed in the respective dropdown. Depending on the size of your "
        "model and chosen hyperparameters, it might take some time. Be aware that if "
        "you change a widget's value or navigate to other pages, the logs dropdown "
        "will disappear. However, you will still be able to examine the history "
        "dataframe and plot the logs in the next section."
    )

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
            with capture.stdout(st.empty().code):
                try:
                    df = data.dataframe

                    model.fit(df, int(batch_size), int(num_epochs), float(val_split))
                    st.toast("Training is completed!", icon="✅")
                except errors.ModelError as error:
                    st.toast(error, icon="❌")


def plot_history_ui(model: model.CreatedModel) -> None:
    """Generate the UI for plotting the training history.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Plot History")
    st.markdown(
        "Plot the training logs by specifying one or more you want to view. You can "
        "also examine and download the training history dataframe from which the "
        "plot is constructed. Additionally, you can interact with the plot by "
        "zooming in and out, dragging it, and accessing different download options "
        "by clicking the three dots in the upper right corner."
    )

    history = model.history
    options = history.columns.drop("epoch") if not history.empty else list()

    with st.form("plot_history_form", border=False):
        y = st.multiselect("Select Y-axis log(s):", options)
        points = st.toggle("Point Markers")
        plot_history_btn = st.form_submit_button("Plot History")

    if plot_history_btn:
        with container.chart_container(history, export_formats=["CSV"]):
            try:
                chart = model.plot_history(y, points)

                st.altair_chart(chart, use_container_width=True)
            except errors.PlotError as error:
                st.toast(error, icon="❌")
