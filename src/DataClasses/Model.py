from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Model:
    """Model container."""

    name: str | None = None
    instance: Any = None
    layers: Dict | None = None
