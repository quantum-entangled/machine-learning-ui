import pytest
import random


@pytest.fixture
def csv_str() -> str:
    column_num = 10
    row_num = 100
    n = 10**5
    random.seed(17)
    csv = ""
    for i in range(column_num):
        csv += f"h{i},"
    csv = csv[:-1] + "\r\n"
    for i in range(row_num):
        for j in range(column_num):
            csv += str(random.uniform(-n, n)) + ","
        csv = csv[:-1] + "\r\n"
    return csv


@pytest.fixture
def empty_csv() -> str:
    return ""


@pytest.fixture
def csv_with_multiindex() -> str:
    return "\r\n".join(["h1,h2", "2,2,5,7", "1,2", "1,1", "3,3", ""])


@pytest.fixture
def csv_with_diff_types() -> str:
    return "\r\n".join(["h1,h2,h3", "1,2,2", "1,1,6", "2,aaaa,5", "3,3,6", ""])


@pytest.fixture
def csv_with_inf() -> str:
    return "\r\n".join(["h1,h2,h3", "1,2,2", "1,1,6", "2,inf,5", "3,3,6", ""])


@pytest.fixture
def csv_with_NaN() -> str:
    return "\r\n".join(["h1,h2,h3", "1,2,2", "1,1,6", "2,,5", "3,3,6", ""])


@pytest.fixture
def csv_invalid_delimiter() -> str:
    return "\r\n".join(["h1:h2:h3", "1:2:2", "1:1:6", "2:7:5", "3:3:6", ""])


@pytest.fixture
def csv_invalid_indentations() -> str:
    return (
        "\r\n".join(["h1,h2,h3", "1,2,2"])
        + "\t"
        + "\r\n".join(["1,1,6", "2,7,5", "3,3,6", ""])
    )
