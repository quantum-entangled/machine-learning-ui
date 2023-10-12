from csv import DictWriter, list_dialects
from io import StringIO
from typing import List, Dict, Callable

from hypothesis.strategies import lists
from hypothesis import strategies

from hypothesis_csv._data_rows import *
from hypothesis_csv.type_utils import *


def _records_to_csv(data: List[Dict], dialect: str, has_header: bool = True):
    f = StringIO()
    w = DictWriter(f, dialect=dialect, fieldnames=data[0].keys())
    if has_header:
        w.writeheader()
    for row in data:
        w.writerow(row)

    return f.getvalue()


def draw_header(draw: Callable, header_len: int):
    return draw(
        lists(
            text(
                min_size=1,
                alphabet=string.ascii_lowercase
                + string.ascii_uppercase
                + string.digits,
            ),
            min_size=header_len,
            max_size=header_len,
            unique=True,
        )
    )


def draw_dialect(draw: Callable):
    return draw(sampled_from(list_dialects()))


@overload
def _get_header_and_column_types(draw: isa(Callable), header, columns):
    raise InvalidArgument("Header or column are of invalid type")


@overload
def _get_header_and_column_types(draw: isa(Callable), header: is_seq, columns: is_seq):
    if len(header) == len(columns):
        return header, columns
    else:
        raise InvalidArgument("Header and columns must be of the same size")


@overload
def _get_header_and_column_types(draw: isa(Callable), header: is_seq, columns: is_none):
    return header, len(header)


@overload
def _get_header_and_column_types(
    draw: isa(Callable), header: is_none, columns: is_none
):
    columns = draw(integers(min_value=1, max_value=10))
    return None, columns


@overload
def _get_header_and_column_types(
    draw: isa(Callable), header: isa(int), columns: isa(int)
):
    if header == columns:
        header_fields = draw_header(draw, header)

        return header_fields, len(header_fields)
    else:
        raise InvalidArgument("Header and columns must be of the same size")


@overload
def _get_header_and_column_types(
    draw: isa(Callable), header: is_none, columns: isa(int)
):
    return None, columns


@overload
def _get_header_and_column_types(
    draw: isa(Callable), header: isa(int), columns: is_none
):
    return _get_header_and_column_types(draw, header, header)


@overload
def _get_header_and_column_types(draw: isa(Callable), header: is_none, columns: is_seq):
    return None, columns


@composite
def csv(
    draw: Callable,
    header: List[str] | int | None = None,
    columns: List[strategies] | int | None = None,
    lines: int | None = None,
    dialect: str | None = "excel",
):
    header_param, columns = _get_header_and_column_types(draw, header, columns)
    rows = list(draw(data_rows(lines=lines, columns=columns)))
    dialect = dialect or draw_dialect(draw)
    header = header_param or ["col_{}".format(i) for i in range(len(rows[0]))]

    data = [dict(zip(header, d)) for d in rows]

    return _records_to_csv(data, has_header=header_param is not None, dialect=dialect)
