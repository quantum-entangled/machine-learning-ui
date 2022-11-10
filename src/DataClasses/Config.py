from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    """Config container."""

    input_training_indices: dict | None
    output_training_indices: dict | None
    optimizer: Any = None
