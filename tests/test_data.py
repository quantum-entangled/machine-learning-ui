import io
import pytest
import pandas as pd
import altair as alt
import mlui.classes.data as data_cls
import mlui.classes.errors as err


class TestUpload:
    def test_upload_data(self, data: data_cls.Data, csv_str: str):
        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            data.upload(buff)
        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        assert data.dataframe.equals(dataframe)

    def test_upload_errors(
        self,
        data: data_cls.Data,
        empty_csv: str,
        csv_with_multiindex: str,
        csv_invalid_delimiter: str,
        csv_invalid_indentations: str,
    ):
        with pytest.raises(err.UploadError):
            with io.BytesIO(bytes(empty_csv, "utf-8")) as buff:
                data.upload(buff)

        with pytest.raises(err.UploadError):
            with io.BytesIO(bytes(csv_with_multiindex, "utf-8")) as buff:
                data.upload(buff)

        with pytest.raises(err.UploadError):
            with io.BytesIO(bytes(csv_invalid_delimiter, "utf-8")) as buff:
                data.upload(buff)

        with pytest.raises(err.UploadError):
            with io.BytesIO(bytes(csv_invalid_indentations, "utf-8")) as buff:
                data.upload(buff)


class TestConfigure:
    def test_set_and_get_unused_columns(self, data: data_cls.Data, csv_str: str):
        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            data.upload(buff)
        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        unused_columns = data.get_unused_columns()
        assert len(unused_columns) == 10

        columns = list(dataframe.columns)
        available, selected = columns[:5], columns[5:]
        data.set_unused_columns(available, selected)

        unused_columns = data.get_unused_columns()
        assert len(unused_columns) == 5
        assert all(unused_columns[i] == available[i] for i in range(5))


class TestData:
    def test_data_info(self, data: data_cls.Data, empty_csv: str, csv_with_NaN: str):
        csv_file = bytes(empty_csv, "utf-8")
        with io.BytesIO(csv_file) as buff:
            data.upload(buff)
        empty = data.empty
        assert empty is True

        csv_file = bytes(csv_with_NaN, "utf-8")
        with io.BytesIO(csv_file) as buff:
            data.upload(buff)
        has_nans = data.has_nans
        assert has_nans is True

    def test_get_stats(self, data: data_cls.Data, csv_str: str):
        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            data.upload(buff)

        stats = data.get_stats()
        assert isinstance(stats, pd.DataFrame)

    def test_plot_columns(
        self, data: data_cls.Data, csv_str: str, csv_with_diff_types: str
    ):
        with pytest.raises(err.PlotError):
            data.plot_columns(None, None, True)

        invalid_csv_file = bytes(csv_with_diff_types, "utf-8")
        with io.BytesIO(invalid_csv_file) as buff:
            data.upload(buff)
        with pytest.raises(err.PlotError):
            data.plot_columns("h1", "h2", True)

        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            data.upload(buff)
        plot = data.plot_columns("h1", "h2", True)
        assert isinstance(plot, alt.Chart)

    def test_reset_state(self, data: data_cls.Data, csv_str: str):
        csv_file = bytes(csv_str, "utf-8")
        with io.BytesIO(csv_file) as buff:
            data.upload(buff)

        with io.BytesIO(csv_file) as buff:
            dataframe = pd.read_csv(buff, header=0, skipinitialspace=True)

        columns = list(dataframe.columns)
        available, selected = columns[:5], columns[5:]
        data.set_unused_columns(available, selected)

        data.reset_state()
        assert data.dataframe.empty is True
        assert len(data.columns) == 0
        unused_columns = data.get_unused_columns()
        assert len(unused_columns) == 0
