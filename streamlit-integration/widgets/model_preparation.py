import io

import data_classes.model as model_cls
import streamlit as st

import enums.layers as layers
import managers.errors as err
import managers.model_manager as mm


def create_model_ui(model: model_cls.Model) -> None:
    """Generate UI for creating a new model.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Create Model")

    model_name = st.text_input(
        "Model name:", max_chars=50, placeholder="Enter Model Name"
    )
    create_model_btn = st.button("Create Model")

    if create_model_btn:
        try:
            mm.create_model(model_name, model)
            st.success("Model is created!", icon="✅")
        except (err.NoModelNameError, err.SameModelNameError) as error:
            st.error(error, icon="❌")


def add_layers_ui(model: model_cls.Model) -> None:
    """Generate UI for managing model's layers.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Add Layers")

    layer = str(st.selectbox("Select layer class:", list(layers.classes)))
    layer_cls = layers.classes[layer]
    layer_widget = layers.widgets[layer](model=model)
    layer_params = layer_widget.params
    layer_connection = layer_widget.connection
    add_layer_btn = st.button("Add Layer")

    if add_layer_btn:
        try:
            mm.add_layer(layer_cls, layer_params, layer_connection, model)
            st.success("Layer is added!", icon="✅")
        except (
            err.NoModelError,
            err.NoLayerNameError,
            err.SameLayerNameError,
            err.NoConnectionError,
        ) as error:
            st.error(error, icon="❌")


def set_outputs_ui(model: model_cls.Model) -> None:
    """Generate UI for setting the model ouputs.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Set Outputs")

    outputs = st.multiselect(
        "Select outputs:", list(set(model.layers) - set(model.input_layers))
    )
    set_outputs_btn = st.button("Set Outputs")

    if set_outputs_btn:
        try:
            mm.set_outputs(outputs, model)
            st.success("Outputs are set!", icon="✅")
        except (err.NoModelError, err.NoOutputsSelectedError) as error:
            st.error(error, icon="❌")


def examine_model(model: model_cls.Model) -> None:
    """Generate UI for examining a model.

    It includes the ability to show the model summary, as well as download the model
    and its graph.

    Parameters
    ----------
    model : Model
        Model container object.
    """
    st.header("Examine Model")

    placeholder = st.empty()

    try:
        with placeholder.container():
            st.markdown("Expand the box below to view the model summary.")

            with st.expander("Model Summary"):
                mm.show_summary(model)

            st.markdown("Click the button below to download the model graph.")
            graph = mm.download_graph(model)

            with io.BytesIO(graph) as graph_pdf_bytes:
                st.download_button("Download Graph", graph_pdf_bytes, "model_graph.pdf")

            st.markdown("Click the button below to download the model.")
            model_object = mm.download_model(model)

            with io.BytesIO(model_object) as model_object_bytes:
                st.download_button("Download Model", model_object_bytes, "model.h5")
    except (err.NoModelError, err.NoOutputLayersError):
        placeholder.info(
            "You will be able to examine the model once you create/upload it and "
            "select the outputs.",
            icon="💡",
        )
