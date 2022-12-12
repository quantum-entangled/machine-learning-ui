from enum import Enum


class Caution(Enum):
    MISSING_VALUES = "\u26a0 Caution: Missing values in columns:"
    NON_NUMERIC = "\u26a0 Caution: Non-numeric values in columns:"

    def __str__(self):
        return f"{self.value}"
