import streamlit as st
import streamlit_extras.capture as capture

import mlui.classes.data as data
import mlui.classes.model as model


def model_info_ui(model: model.Model) -> None:
    """Generate the UI for displaying the information about the model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Model Info")
    st.markdown(
        "View the essential information about your model in relation to other sections."
    )

    task = st.session_state.get("task")
    built = model.built
    input_configured = model.input_configured
    output_configured = model.output_configured
    compiled = model.compiled

    if not built:
        st.info("The model is not uploaded/created.", icon="💡")
        return

    if input_configured:
        st.success("The model's input layers are configured.", icon="✅")
    else:
        st.info("The model's input layers are not configured.", icon="💡")

    if task != "Predict" and output_configured:
        st.success("The model's output layers are configured.", icon="✅")
    elif task != "Predict" and not output_configured:
        st.info("The model's output layers are not configured.", icon="💡")

    if task != "Predict" and compiled:
        st.success("The model is compiled.", icon="✅")
    elif task != "Predict" and not compiled:
        st.info("The model is not compiled.", icon="💡")


def summary_ui(model: model.Model) -> None:
    """Generate the UI for displaying the summary of the model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Summary")
    st.markdown("View the model's summary.")

    with st.expander("Summary"):
        with capture.stdout(st.empty().code):
            model.summary


def graph_ui(model: model.Model) -> None:
    """Generate the UI for downloading the graph of the model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Graph")
    st.markdown("Download the graph describing the model's architecture.")

    name = model.name
    graph = model.graph

    st.download_button("Download Graph", graph, f"{name}_graph.pdf")


def download_model_ui(model: model.Model) -> None:
    """Generate the UI for downloading the model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Download Model")
    st.markdown("Download the model in `H5` format.")

    name = model.name
    model_as_bytes = model.as_bytes

    st.download_button("Download Model", model_as_bytes, f"{name}.h5")


def reset_model_ui(data: data.Data, model: model.Model) -> None:
    """Generate the UI for resetting the model.

    Parameters
    ----------
    data : Data
        Data object.
    model : Model
        Model object.
    """
    st.header("Reset Model")
    st.markdown(
        "Reset your model for the app session. Once the `Reset Model` button is "
        "clicked, the model file you have uploaded/created will be erased, and all "
        "connections with other sections will be removed."
    )

    reset_model_btn = st.button("Reset Model")

    if reset_model_btn:
        model.reset_state()
        data.update_state()
        st.rerun()
