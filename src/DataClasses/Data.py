from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Data:
    """Data container."""

    file: pd.DataFrame = pd.DataFrame()
    columns: list[str] = field(default_factory=list)
    input_columns: dict[str, list[str]] = field(default_factory=dict)
    output_columns: dict[str, list[str]] = field(default_factory=dict)
    columns_per_layer: dict[str, int] = field(default_factory=dict)
    input_train_data: dict[str, np.ndarray] = field(default_factory=dict)
    output_train_data: dict[str, np.ndarray] = field(default_factory=dict)
    input_test_data: dict[str, np.ndarray] = field(default_factory=dict)
    output_test_data: dict[str, np.ndarray] = field(default_factory=dict)
