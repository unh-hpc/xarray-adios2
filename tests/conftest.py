from __future__ import annotations

import adios2py
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def sample_dataset():
    return xr.Dataset(
        {"arr1d": (("time", "x"), np.arange(15.0).reshape(3, 5))},
        coords={"x": np.arange(5.0), "time": np.arange(3.0)},
    )


@pytest.fixture
def test_file(tmp_path, sample_dataset):
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
