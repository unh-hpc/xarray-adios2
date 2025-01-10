from __future__ import annotations

import adios2py
import numpy as np

from xarray_adios2 import Adios2Store


def test_ctor(test_file):
    with adios2py.File(test_file, mode="rra") as file:
        store = Adios2Store(file)
        assert store.ds.keys() == {"arr1d", "x", "time"}


def test_open(test_file):
    store = Adios2Store.open(test_file, mode="rra")
    assert store.ds.keys() == {"arr1d", "x", "time"}


def test_load(test_file, sample_dataset):
    with adios2py.File(test_file, mode="rra") as file:
        with file.steps.next() as step:
            store = Adios2Store(step)
            vars, attrs = store.load()  # type: ignore[no-untyped-call]
            for name in sample_dataset.variables:
                assert vars[name].sizes == sample_dataset[name].sizes
                assert np.array_equal(vars[name], sample_dataset[name])
        assert attrs == sample_dataset.attrs
