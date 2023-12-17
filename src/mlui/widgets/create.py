import streamlit as st

import mlui.classes.errors as errors
import mlui.classes.model as model
import mlui.enums as enums


def set_name_ui(model: model.CreatedModel) -> None:
    """Generate the UI for setting the model's name.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Name")
    st.markdown(
        "Specify a name for the model to be created. If no name is provided, the "
        "default value will be used."
    )

    default = model.name

    name = st.text_input("Enter model's name:", max_chars=50, placeholder=default)

    def set_name() -> None:
        """Supporting function for the accurate representation of widgets."""
        try:
            model.set_name(name if name else default)
            st.toast("Name is set!", icon="✅")
        except errors.SetError as error:
            st.toast(error, icon="❌")

    st.button("Set Name", on_click=set_name)


def set_layers_ui(model: model.CreatedModel) -> None:
    """Generate the UI for setting the model's layers.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Layers")
    st.markdown(
        "Add layers to the model, specifying their class, name (default value if "
        "nothing is provided), and additional parameters. Always start with `Input` "
        "layer(s). If needed, you can delete the last added layer in case of a "
        "mistake or if you want to make changes."
    )

    layers = enums.layers.classes
    objects = model.layers
    default = f"layer_{len(objects) + 1}"

    entity = str(st.selectbox("Select layer's class:", layers))
    name = st.text_input("Enter layer's name:", max_chars=50, placeholder=default)

    with st.expander("Layer's Parameters"):
        prototype = enums.layers.widgets[entity]
        widget = prototype(objects)

    def set_layer() -> None:
        """Supporting function for the accurate representation of widgets."""
        try:
            params = widget.params
            connection = widget.get_connection()

            model.set_layer(entity, name if name else default, params, connection)
            st.toast("Layer is set!", icon="✅")
        except (errors.LayerError, errors.SetError) as error:
            st.toast(error, icon="❌")

    def delete_last_layer() -> None:
        """Supporting function for the accurate representation of widgets."""
        try:
            model.delete_last_layer()
            st.toast("Last layer is deleted!", icon="✅")
        except errors.DeleteError as error:
            st.toast(error, icon="❌")

    st.button("Set Layer", on_click=set_layer)
    st.button("Delete Last Layer", on_click=delete_last_layer)


def set_outputs_ui(model: model.CreatedModel) -> None:
    """Generate the UI for setting the model's outputs.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Outputs")
    st.markdown(
        "Specify the model's outputs among the added layers. `Input` layers are "
        "not included here."
    )

    layers = set(model.layers) - set(model.inputs)
    default = model.outputs

    outputs = st.multiselect("Select outputs:", layers, default)

    def set_outputs() -> None:
        """Supporting function for the accurate representation of widgets."""
        try:
            model.set_outputs(outputs)
            st.toast("Outputs are set!", icon="✅")
        except errors.SetError as error:
            st.toast(error, icon="❌")

    st.button("Set Outputs", on_click=set_outputs)


def create_model_ui(model: model.CreatedModel) -> None:
    """Generate the UI for building the model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Create Model")
    st.markdown(
        "By pressing the provided button, you will create a model with the parameters "
        "specified above. Any changes made to the previous sections afterward will "
        "not take effect until you click the button again."
    )

    def build_model() -> None:
        """Supporting function for the accurate representation of widgets."""
        try:
            model.create()
            st.toast("Model is created!", icon="✅")
        except errors.CreateError as error:
            st.toast(error, icon="❌")

    st.button("Create Model", on_click=build_model)
