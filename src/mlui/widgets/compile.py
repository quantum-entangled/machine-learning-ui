import streamlit as st

from mlui.classes.errors import ModelError, SetError
from mlui.classes.model import Model
from mlui.enums import losses, metrics, optimizers


def set_optimizer_ui(model: Model) -> None:
    """Generate the UI for setting the model's optimizer.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Optimizer")

    options = list(optimizers.classes)
    current_optimizer = model.get_optimizer()
    index = options.index(current_optimizer) if current_optimizer else 0
    entity = str(st.selectbox("Select optimizer's class:", options, index))

    with st.expander("Optimizer's Parameters"):
        widget = optimizers.widgets[entity]()

    def set_optimizer() -> None:
        try:
            params = widget.params

            model.set_optimizer(entity, params)
            st.toast("Optimizer is set!", icon="✅")
        except SetError as error:
            st.toast(error, icon="❌")

    st.button("Set Optimizer", on_click=set_optimizer)


def set_loss_functions_ui(model: Model) -> None:
    """Generate the UI for setting the model's loss functions.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Loss Functions")

    layer_options = model.outputs
    proto_options = losses.classes
    layer = str(st.selectbox("Select layer:", layer_options, key="losses"))
    current_loss = model.get_loss(layer)
    index = proto_options.index(current_loss) if current_loss else 0
    entity = str(st.selectbox("Select loss function's class:", proto_options, index))

    def set_loss() -> None:
        model.set_loss(layer, entity)
        st.toast("Loss is set!", icon="✅")

    st.button("Set Loss Function", on_click=set_loss)


def set_metrics_ui(model: Model) -> None:
    """Generate the UI for setting the model's metrics.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Set Metrics")

    layer_options = model.outputs
    layer = str(st.selectbox("Select layer:", layer_options, key="metrics"))
    current_metrics = model.get_metrics(layer)
    options = metrics.classes
    entities = st.multiselect("Select metrics:", options, current_metrics)

    def set_metrics() -> None:
        model.set_metrics(layer, entities)
        st.toast("Metrics are set!", icon="✅")

    st.button("Set Metrics", on_click=set_metrics)


def compile_model_ui(model: Model) -> None:
    """Generate the UI for compiling the model.

    Parameters
    ----------
    model : Model
        Model object.
    """
    st.header("Compile Model")

    def compile_model() -> None:
        try:
            model.compile()
            st.toast("Model is compiled!", icon="✅")
        except ModelError as error:
            st.toast(error, icon="❌")

    st.button("Compile Model", on_click=compile_model)
