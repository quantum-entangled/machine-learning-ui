import streamlit as st

from ..data_classes import data as data_cls
from ..data_classes import model as model_cls
from ..managers import errors as err
from ..managers import model_manager as mm


def evaluate_model_ui(data: data_cls.Data, model: model_cls.Model) -> None:
    """Generate UI for evaluating the model.

    Parameters
    ----------
    data : Data
        Data container object.
    model : Model
        Model container object.
    """
    st.header("Evaluate Model")

    batch_size = int(
        st.number_input("Batch size:", min_value=1, max_value=1024, value=32, step=1)
    )
    evaluate_model_btn = st.button("Evaluate Model")

    if evaluate_model_btn:
        try:
            results = mm.evaluate_model(batch_size, data, model)
            st.write("Results:", results)
            st.success("Evaluation is completed!", icon="✅")
        except (
            err.NoModelError,
            err.DataNotSplitError,
            err.ModelNotCompiledError,
        ) as error:
            st.error(error, icon="❌")
