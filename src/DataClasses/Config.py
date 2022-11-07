from dataclasses import dataclass


@dataclass
class Config:
    """Config container."""

    input_training_indices: dict | None
    output_training_indices: dict | None
