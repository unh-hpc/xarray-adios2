from __future__ import annotations

import adios2py
import numpy as np
import pytest
import xarray as xr

from xarray_adios2 import Adios2Store


def test_open_one_step_adios2py(one_step_file_adios2py, sample_dataset):
    with adios2py.File(one_step_file_adios2py, mode="rra") as file:  # noqa: SIM117
        with file.steps.next() as step:
            with xr.open_dataset(Adios2Store(step)) as ds:
                assert ds == sample_dataset


def test_open_one_step(one_step_file, sample_dataset):
    with adios2py.File(one_step_file, mode="rra") as file:  # noqa: SIM117
        with file.steps.next() as step:
            with xr.open_dataset(Adios2Store(step)) as ds:
                assert ds == sample_dataset


@pytest.fixture
def test_by_step_file_adios2py(tmp_path, sample_dataset):
    filename = tmp_path / "test1.bp"
    with adios2py.File(filename, mode="w") as file:
        for time, step in zip(sample_dataset["time"], file.steps, strict=False):
            ds_step = sample_dataset.sel(time=time)
            for name in ds_step.variables:
                step[name] = ds_step[name]
                step[name].attrs["dimensions"] = " ".join(ds_step[name].dims)

    return filename


def test_open_by_step(test_by_step_file_adios2py, sample_dataset):
    with xr.open_dataset(test_by_step_file_adios2py, step_dimension="time") as ds:
        assert ds.keys() == sample_dataset.keys()
        assert ds.sizes == sample_dataset.sizes
        assert ds.coords.keys() == sample_dataset.coords.keys()
        assert ds.broadcast_equals(sample_dataset)
        for name in sample_dataset.variables:
            if name == "x":
                assert np.array_equal(ds[name].isel(time=0), sample_dataset[name])
            else:
                assert np.array_equal(ds[name], sample_dataset[name])
