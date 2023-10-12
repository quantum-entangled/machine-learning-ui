from multimethod import isa
import collections
from collections.abc import Iterable
from typing import Any


def is_none(x: Any):
    return x is None


def is_seq(x: Iterable | str):
    return isa(Iterable)(x) and not isa(str)(x)
