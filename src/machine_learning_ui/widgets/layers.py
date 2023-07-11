import abc
from typing import Callable, TypeAlias, TypedDict

import data_classes.model as model_cls
import streamlit as st
import tensorflow as tf
from enums.activations import activations

LayerConnection: TypeAlias = str | list[str] | int | None


class LayerParams(TypedDict):
    """Base type annotation for a layer's parameters."""

    name: str


class InputParams(LayerParams):
    """Type annotation for the Input layer."""

    shape: tuple[int]


class DenseParams(LayerParams):
    """Type annotation for the Dense layer."""

    units: int
    activation: Callable[..., tf.Tensor]


class LayerWidget(abc.ABC):
    """Base class for a layer's widget."""

    @abc.abstractmethod
    def __init__(self, model: model_cls.Model) -> None:
        """
        Parameters
        ----------
        model : Model
            Model container object.
        """
        self.model = model

    @property
    @abc.abstractmethod
    def params(self) -> LayerParams:
        """Layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """

    @property
    @abc.abstractmethod
    def connection(self) -> LayerConnection:
        """Connection layer.

        Returns
        -------
        str, list of str, int, or None
            Reference to a connection layer in a model.
        """


class Input(LayerWidget):
    """Input layer widget."""

    def __init__(self, model: model_cls.Model) -> None:
        """
        Parameters
        ----------
        model : Model
            Model container object.
        """
        self.model = model
        self.layer_name = str(
            st.text_input("Layer name:", max_chars=50, placeholder="Enter Layer Name")
        )
        self.input_shape = int(
            st.number_input(
                "Number of input columns:", value=1, min_value=1, max_value=10_000
            )
        )

    @property
    def params(self) -> InputParams:
        """Layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {"name": self.layer_name, "shape": (self.input_shape,)}

    @property
    def connection(self) -> None:
        """Connection layer."""
        return


class Dense(LayerWidget):
    """Dense layer widget."""

    def __init__(self, model: model_cls.Model) -> None:
        """
        Parameters
        ----------
        model : Model
            Model container object.
        """
        self.model = model
        self.layer_name = str(
            st.text_input("Layer name:", max_chars=50, placeholder="Enter Layer Name")
        )
        self.units_num = int(
            st.number_input("Number of units:", value=1, min_value=1, max_value=10_000)
        )

        self.activation = st.selectbox("Activation function:", list(activations))
        self.connect_to = st.selectbox("Connect layer to:", list(self.model.layers))

    @property
    def params(self) -> DenseParams:
        """Layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {
            "name": self.layer_name,
            "units": self.units_num,
            "activation": activations[self.activation]
            if self.activation
            else activations["Linear"],
        }

    @property
    def connection(self) -> str | int:
        """Connection layer.

        Returns
        -------
        str or int
            Layer name if selected, 0 otherwise.
        """
        return self.connect_to if self.connect_to else 0


class Concatenate(LayerWidget):
    """Concatenate layer widget."""

    def __init__(self, model: model_cls.Model) -> None:
        """
        Parameters
        ----------
        model : Model
            Model container object.
        """
        self.model = model
        self.layer_name = str(
            st.text_input("Layer name:", max_chars=50, placeholder="Enter Layer Name")
        )
        self.concatenate = st.multiselect(
            "Select layers (at least 2):", options=list(self.model.layers)
        )

    @property
    def params(self) -> LayerParams:
        """Layer's parameters.

        Returns
        -------
        dict
            Dictionary containing values of adjustable parameters.
        """
        return {"name": self.layer_name}

    @property
    def connection(self) -> list[str] | int:
        """Connection layers.

        Returns
        -------
        list of str or int
            List of layers' names if selected, 0 otherwise.
        """
        return (
            [layer_name for layer_name in self.concatenate]
            if len(self.concatenate) >= 2
            else 0
        )
