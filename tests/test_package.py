from __future__ import annotations

import importlib.metadata

import xarray_adios2 as m


def test_version():
    assert importlib.metadata.version("xarray_adios2") == m.__version__
