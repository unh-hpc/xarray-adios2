from __future__ import annotations

import adios2py
import numpy as np
import pytest

from xarray_adios2 import Adios2Store


def test_ctor(one_step_file_adios2py):
    with adios2py.File(one_step_file_adios2py, mode="rra") as file:
        store = Adios2Store(file)
        assert store.ds.keys() == {"arr1d", "x", "time"}


def test_open(one_step_file_adios2py):
    store = Adios2Store.open(one_step_file_adios2py, mode="rra")
    assert store.ds.keys() == {"arr1d", "x", "time"}


@pytest.mark.parametrize("filename", ["one_step_file_adios2py", "one_step_file"])
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


def test_one_step_file(one_step_file, sample_dataset):
    with adios2py.File(one_step_file, mode="rra") as file:  # noqa: SIM117
        with file.steps.next() as step:
            store = Adios2Store(step)
            vars, attrs = store.load()  # type: ignore[no-untyped-call]
            for name in sample_dataset.variables:
                assert vars[name].sizes == sample_dataset[name].sizes
                assert np.array_equal(vars[name], sample_dataset[name])
            assert attrs == sample_dataset.attrs
