from __future__ import annotations

import adios2py
import numpy as np
import pytest

from xarray_adios2 import Adios2Store


@pytest.fixture
def test_file_store(tmp_path, sample_dataset):
    ds = sample_dataset
    filename = tmp_path / "test.bp"
    with adios2py.File(filename, mode="w") as file:  # noqa: SIM117
        with file.steps.next() as step:
            for name, var in ds.variables.items():
                step[name] = var
                step[name].attrs["dimensions"] = " ".join(var.dims)
                step[name].attrs["dtype"] = str(var.dtype)
                for attr_name, attr in var.attrs.items():
                    step[name].attrs[attr_name] = attr

    return filename


def test_ctor(test_file):
    with adios2py.File(test_file, mode="rra") as file:
        store = Adios2Store(file)
        assert store.ds.keys() == {"arr1d", "x", "time"}


def test_open(test_file):
    store = Adios2Store.open(test_file, mode="rra")
    assert store.ds.keys() == {"arr1d", "x", "time"}


@pytest.mark.parametrize("filename", ["test_file", "test_file_store"])
def test_load(filename, sample_dataset, request):
    filename = request.getfixturevalue(filename)
    with adios2py.File(filename, mode="rra") as file:
        with file.steps.next() as step:
            store = Adios2Store(step)
            vars, attrs = store.load()  # type: ignore[no-untyped-call]
            for name in sample_dataset.variables:
                assert vars[name].sizes == sample_dataset[name].sizes
                assert np.array_equal(vars[name], sample_dataset[name])
        assert attrs == sample_dataset.attrs
