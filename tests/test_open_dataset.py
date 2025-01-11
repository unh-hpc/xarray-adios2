from __future__ import annotations

import adios2py
import numpy as np
import pytest
import xarray as xr

from xarray_adios2 import Adios2Store


def test_open(test_file, sample_dataset):
    with adios2py.File(test_file, mode="rra") as file:  # noqa: SIM117
        with file.steps.next() as step:
            with xr.open_dataset(Adios2Store(step)) as ds:
                assert ds == sample_dataset


@pytest.fixture
def test_by_step_file(tmp_path):
    filename = tmp_path / "test1.bp"
    with adios2py.File(filename, mode="w") as file:
        for n, step in zip(range(5), file.steps, strict=False):
            step["time"] = 10.0 * n
            step["time"].attrs["dimensions"] = "time"

            step["x"] = np.linspace(0, 1, 10)
            step["x"].attrs["dimensions"] = "redundant x"

            step["arr1d"] = np.arange(10) + n
            step["arr1d"].attrs["dimensions"] = "time x"

    return filename


def test_open_by_step(test_by_step_file):
    with xr.open_dataset(test_by_step_file) as ds:
        assert ds.keys() == {"arr1d"}
        assert ds.sizes == {"time": 5, "x": 10, "redundant": 5}
        assert ds.coords.keys() == {"time", "x"}
        assert np.array_equal(ds.arr1d.isel(time=0), np.arange(10))
        assert np.array_equal(ds.arr1d.isel(time=1), np.arange(10) + 1)
