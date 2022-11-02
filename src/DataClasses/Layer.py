from dataclasses import dataclass
from typing import Any


@dataclass
class Layer:
    """Layer container."""

    instance: Any = None
    widget: Any = None
