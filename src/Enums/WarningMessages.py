from enum import Enum


class Warnings(Enum):
    MISSING_VALUES = '\u26a0 Warning: missing values in columns:'
    NON_NUMERIC = '\u26a0 Warning: non numeric values in columns:'

    def __str__(self):
        return f"{self.value}"
