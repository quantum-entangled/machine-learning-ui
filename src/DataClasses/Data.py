from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Data:
    """Data container."""

    file: pd.DataFrame = pd.DataFrame()
    columns: list[str] = field(default_factory=list)
    input_columns: dict[str, list[str]] = field(default_factory=dict)
    output_columns: dict[str, list[str]] = field(default_factory=dict)
    columns_per_layer: dict[str, int] = field(default_factory=dict)
    input_training_data: dict[str, Any] = field(default_factory=dict)
    output_training_data: dict[str, Any] = field(default_factory=dict)
    input_test_data: dict[str, Any] = field(default_factory=dict)
    output_test_data: dict[str, Any] = field(default_factory=dict)
