import streamlit as st

import mlui.classes.data as data
import mlui.classes.errors as errors
import mlui.classes.model as model
import mlui.enums as enums


def set_features_ui(data: data.Data, model: model.Model) -> None:
    """Generate the UI for setting the data columns as features for the input and
    output layers of the model.

    Parameters
    ----------
    data : Data
        Data object.
    model : Model
        Model object.
    """
    st.header("Set Input/Output Features")
    st.markdown(
        "Choose input and output features for each respective layer from the data "
        "columns. Please be aware that the order in which you add the columns is "
        "important for evaluating the model or making predictions, as the data needs "
        "to be consistent with the data on which the model was trained. Additionally, "
        "note that for multiclass classification problems, the output columns should "
        "be one-hot encoded for the model to work correctly."
    )

    task = st.session_state.get("task")

    if task != "Predict":
        side = st.selectbox("Select layer's type:", ("Input", "Output"))
    else:
        side = st.selectbox("Select layer's type:", ("Input",))

    if side == "Input":
        at = "input"
        layers = model.inputs
        shapes = model.input_shape
    else:
        at = "output"
        layers = model.outputs
        shapes = model.output_shape

    layer = str(st.selectbox("Select layer:", layers))
    available = data.get_unused_columns()
    default = model.get_features(layer, at)
    available.extend(default)
    columns = st.multiselect(
        "Select columns (order is important):",
        available,
        default,
        max_selections=shapes[layer],
    )

    def set_features() -> None:
        """Supporting function for the accurate representation of widgets."""
        try:
            model.set_features(layer, columns, at)
            data.set_unused_columns(available, columns)
            st.toast("Features are set!", icon="✅")
        except errors.SetError as error:
            st.toast(error, icon="❌")

    st.button("Set Features", on_click=set_features)


def set_callbacks_ui(model: model.Model) -> None:
    """Generate the UI for setting the callbacks for the model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Callbacks")
    st.markdown(
        "Optionally choose callbacks for the model to use during evaluation, training, "
        "or making predictions. Some callbacks have adjustable parameters. Once you "
        "add a callback, you may delete it if you no longer need it or want to "
        "readjust its parameters."
    )

    callbacks = enums.callbacks.classes
    entity = str(st.selectbox("Select callback's class:", callbacks))

    with st.expander("Callback's Parameters"):
        prototype = enums.callbacks.widgets[entity]
        widget = prototype()

    is_set = model.get_callback(entity)
    label = "Set Callback" if not is_set else "Delete Callback"

    def manage_callback() -> None:
        """Supporting function for the accurate representation of widgets."""
        if not is_set:
            try:
                params = widget.params

                model.set_callback(entity, params)
                st.toast("Callback is set!", icon="✅")
            except errors.SetError as error:
                st.toast(error, icon="❌")
        elif is_set:
            model.delete_callback(entity)
            st.toast("Callback is deleted!", icon="✅")

    st.button(label, on_click=manage_callback)
