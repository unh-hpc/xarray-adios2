from __future__ import annotations

import adios2py
import numpy as np
import pytest

from xarray_adios2 import Adios2Store


@pytest.fixture
def test1_file(tmp_path):
    filename = tmp_path / "test1.bp"
    with adios2py.File(filename, mode="w") as file:
        for n, step in zip(range(5), file.steps, strict=False):
            step["step"] = n
            step["time"] = 10.0 * n

            step["x"] = np.linspace(0, 1, 10)
            step["x"].attrs["dimensions"] = "x"

            step["arr1d"] = np.arange(10)
            step["arr1d"].attrs["dimensions"] = "x"

    return filename


def test_ctor(test1_file):
    with adios2py.File(test1_file, mode="rra") as file:
        store = Adios2Store(file)
        assert store.ds.keys() == {"arr1d", "x", "step", "time"}


def test_open(test1_file):
    store = Adios2Store.open(test1_file, mode="rra")
    assert store.ds.keys() == {"arr1d", "x", "step", "time"}
