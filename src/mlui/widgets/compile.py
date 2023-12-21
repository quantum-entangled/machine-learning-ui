import streamlit as st

import mlui.classes.errors as errors
import mlui.classes.model as model
import mlui.enums as enums


def set_optimizer_ui(model: model.Model) -> None:
    """Generate the UI for setting the model's optimizer.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Optimizer")
    st.markdown(
        "Choose an optimizer for model to use during training and adjust its "
        "parameters if needed."
    )

    optimizers = list(enums.optimizers.classes)
    optimizer = model.get_optimizer()
    default = optimizers.index(optimizer) if optimizer else 0

    entity = str(st.selectbox("Select optimizer's class:", optimizers, default))

    with st.expander("Optimizer's Parameters"):
        prototype = enums.optimizers.widgets[entity]
        widget = prototype()

    def set_optimizer() -> None:
        """Supporting function for the accurate representation of widgets."""
        try:
            params = widget.params

            model.set_optimizer(entity, params)
            st.toast("Optimizer is set!", icon="✅")
        except errors.SetError as error:
            st.toast(error, icon="❌")

    st.button("Set Optimizer", on_click=set_optimizer)


def set_loss_functions_ui(model: model.Model) -> None:
    """Generate the UI for setting the model's loss functions.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Loss Functions")
    st.markdown(
        "Choose a loss function for each respective output layer of the model to use "
        "during evaluation or training. Select `BinaryCrossentropy` loss for "
        "binary classification, `CategoricalCrossentropy` for multiclass "
        "classification, and any other for regression."
    )

    layers = model.outputs
    losses = enums.losses.classes

    layer = str(st.selectbox("Select layer:", layers, key="losses"))

    loss = model.get_loss(layer)
    default = losses.index(loss) if loss else 0

    entity = str(st.selectbox("Select loss function's class:", losses, default))

    def set_loss() -> None:
        """Supporting function for the accurate representation of widgets."""
        model.set_loss(layer, entity)
        st.toast("Loss is set!", icon="✅")

    st.button("Set Loss Function", on_click=set_loss)


def set_metrics_ui(model: model.Model) -> None:
    """Generate the UI for setting the model's metrics.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Metrics")
    st.markdown(
        "Optionally choose additional metrics for the model to track during "
        "evaluation or training. You may select multiple metrics for each layer. "
        "`Crossentropy` and `Accuracy` metrics are automatically adjusted to the "
        "type of classification problem you solve."
    )

    layers = model.outputs
    metrics = enums.metrics.classes

    layer = str(st.selectbox("Select layer:", layers, key="metrics"))

    default = model.get_metrics(layer)

    entities = st.multiselect("Select metrics:", metrics, default)

    def set_metrics() -> None:
        """Supporting function for the accurate representation of widgets."""
        model.set_metrics(layer, entities)
        st.toast("Metrics are set!", icon="✅")

    st.button("Set Metrics", on_click=set_metrics)


def compile_model_ui(model: model.Model) -> None:
    """Generate the UI for compiling the model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Compile Model")
    st.markdown(
        "Compile the model by pressing the provided button. It will use the "
        "parameters selected above. Any changes made to the previous sections "
        "afterward will not take effect until you click the button again."
    )

    def compile_model() -> None:
        """Supporting function for the accurate representation of widgets."""
        try:
            model.compile()
            st.toast("Model is compiled!", icon="✅")
        except errors.ModelError as error:
            st.toast(error, icon="❌")

    st.button("Compile Model", on_click=compile_model)
