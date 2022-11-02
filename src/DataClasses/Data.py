from dataclasses import dataclass
from typing import Any


@dataclass
class Data:
    """Data container."""

    file: Any = None
    headers: Any = None
