from collections import Counter
import itertools as it

import streamlit as st

if "original_columns" not in st.session_state:
    st.session_state.original_columns = ["col1", "col2", "col3", "col4"]

if "input_columns" not in st.session_state:
    st.session_state.input_columns = {
        "input_1": list(),
        "input_2": list(),
        "input_3": list(),
    }

if "available_columns" not in st.session_state:
    st.session_state.available_columns = st.session_state.original_columns.copy()


def add_columns() -> None:
    st.session_state.input_columns[layer] = columns
    counter_original = Counter(st.session_state.original_columns)
    counter_input = Counter(
        it.chain.from_iterable(st.session_state.input_columns.values())
    )
    st.session_state.available_columns = [
        item for item, count in counter_original.items() if count > counter_input[item]
    ]


layer = st.selectbox(
    "Choose layer:", ("input_1", "input_2", "input_3"), on_change=add_columns
)

if layer:
    layer_columns = st.session_state.input_columns[layer]
    columns = st.multiselect(
        "Select columns for this layer:",
        st.session_state.available_columns + layer_columns,
        layer_columns,
    )
