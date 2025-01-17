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
def by_step_file_adios2py(tmp_path, sample_dataset):
    filename = tmp_path / "by_step_adios2py.bp"
    step_dimension = "time"
    with adios2py.File(filename, mode="w") as file:
        file.attrs["step_dimension"] = step_dimension
        for time, step in zip(sample_dataset[step_dimension], file.steps, strict=False):
            ds_step = sample_dataset.sel({step_dimension: time})
            for name in ds_step.variables:
                step[name] = ds_step[name]
                step[name].attrs["dimensions"] = " ".join(ds_step[name].dims)

    return filename


@pytest.fixture
def by_step_file(tmp_path, sample_dataset):
    filename = tmp_path / "by_step.bp"
    step_dimension = "time"
    with adios2py.File(filename, mode="w") as file:
        file.attrs["step_dimension"] = step_dimension
        for time, step in zip(sample_dataset[step_dimension], file.steps, strict=False):
            store = Adios2Store(step)
            ds_step = sample_dataset.sel({step_dimension: time})
            ds_step.dump_to_store(store)

    return filename


@pytest.fixture
def by_step_file_at_once(tmp_path, sample_dataset):
    filename = tmp_path / "by_step_at_once.bp"
    # FIXME, should be put into .encoding, but it's lost by the time we get to
    # Adios2Store.store
    sample_dataset.attrs["step_dimension"] = "time"
    with adios2py.File(filename, mode="w") as file:
        sample_dataset.dump_to_store(Adios2Store(file))

    return filename


@pytest.mark.parametrize(
    "filename", ["by_step_file_adios2py", "by_step_file", "by_step_file_at_once"]
)
def test_open_by_step(filename, sample_dataset, request):
    filename = request.getfixturevalue(filename)
    with xr.open_dataset(filename) as ds:
        assert ds.keys() == sample_dataset.keys()
        assert ds.sizes == sample_dataset.sizes
        assert ds.coords.keys() == sample_dataset.coords.keys()
        for name, coord in ds.coords.items():
            assert coord.dims == (name,)
        assert ds.broadcast_equals(sample_dataset)
        for name in sample_dataset.variables:
            assert np.array_equal(ds[name], sample_dataset[name])


@pytest.mark.parametrize(
    "filename", ["by_step_file_adios2py", "by_step_file", "by_step_file_at_once"]
)
def test_open_by_step_streaming(filename, sample_dataset, request):
    filename = request.getfixturevalue(filename)
    with adios2py.File(filename, "r") as file:
        for n, step in enumerate(file.steps):
            ds_step = xr.open_dataset(Adios2Store(step))
            ds_step = ds_step.set_coords("time")
            sample_step = sample_dataset.isel(time=n)
            assert ds_step.equals(sample_step)
            assert set(ds_step.coords.keys()) == set(sample_step.coords.keys())
