from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Protocol

import adios2py
from typing_extensions import Never, override
from xarray.backends import CachingFileManager, DummyFileManager, FileManager
from xarray.backends.common import (
    WritableCFDataStore,
)
from xarray.backends.locks import (
    SerializableLock,
    combine_locks,
    ensure_lock,
    get_write_lock,
)
from xarray.core import indexing
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

from .adios2array import Adios2Array

# adios2 is not thread safe
ADIOS2_LOCK = SerializableLock()


class Lock(Protocol):
    """Provides duck typing for xarray locks, which do not inherit from a common base class."""

    def acquire(self, blocking: bool = True) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
    def locked(self) -> bool: ...


class Adios2Store(WritableCFDataStore):
    """DataStore to facilitate loading an Adios2 file."""

    def __init__(
        self,
        manager: FileManager | adios2py.Group,
        mode: str | None = None,
        lock: Lock = ADIOS2_LOCK,
        autoclose: bool = False,
        **kwargs: Any,
    ):
        if isinstance(manager, adios2py.Group):
            mode = manager._file._mode
            manager = DummyFileManager(manager)  # type: ignore[no-untyped-call]

        assert isinstance(manager, FileManager)
        self._manager = manager
        self._mode = mode
        self.lock = ensure_lock(lock)  # type: ignore[no-untyped-call]
        self.autoclose = autoclose
        self._step_dimension = kwargs.pop("step_dimension", None)
        self._filename = self.ds._file.filename

    @classmethod
    def open(
        cls,
        filename: str | os.PathLike[Any],
        mode: str = "rra",
        lock: Lock | None = None,
        autoclose: bool = False,
        **kwargs: Any,
    ) -> Adios2Store:
        if lock is None:
            if mode in ("r", "rra"):
                lock = ADIOS2_LOCK
            else:
                lock = combine_locks([ADIOS2_LOCK, get_write_lock(filename)])  # type: ignore[no-untyped-call]

        manager = CachingFileManager(adios2py.File, filename, mode=mode)
        return cls(manager, mode=mode, lock=lock, autoclose=autoclose, **kwargs)

    def acquire(self, needs_lock: bool = True) -> adios2py.Group:
        with self._manager.acquire_context(needs_lock) as group:  # type: ignore[no-untyped-call]
            ds = group
        assert isinstance(ds, adios2py.Group)
        return ds

    @property
    def ds(self) -> adios2py.Group:
        return self.acquire()

    def open_store_variable(self, name: str, var: adios2py.ArrayProxy) -> Variable:
        attrs = dict(var.attrs)
        dimensions = attrs.pop("dimensions", "").split()
        dimensions = (
            [self._step_dimension, *dimensions] if self._step_dimension else dimensions
        )
        data = indexing.LazilyIndexedArray(Adios2Array(name, self))
        encoding: dict[str, Any] = {}

        # save source so __repr__ can detect if it's local or not
        encoding["source"] = self._filename
        encoding["original_shape"] = var.shape
        encoding["dtype"] = var.dtype

        return Variable(dimensions, data, attrs, encoding)

    @override
    def get_variables(self) -> Mapping[str, Variable]:
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.items()
        )

    @override
    def get_attrs(self) -> Mapping[str, Any]:
        return FrozenDict(self.ds.attrs)

    @override
    def get_dimensions(self) -> Never:
        raise NotImplementedError()

    @override
    def store(
        self,
        variables: Mapping[str, Variable],
        attributes: Mapping[str, Any],
        check_encoding_set: Any = frozenset(),
        writer: Any = None,
        unlimited_dims: bool | None = None,
    ) -> None:
        variables, attributes = self.encode(variables, attributes)  # type:ignore[no-untyped-call]

        if isinstance(self.ds, adios2py.File):
            with self.ds.steps.next() as step:
                self._write(step, variables, attributes)
        elif isinstance(self.ds, adios2py.Step):
            self._write(self.ds, variables, attributes)
        else:
            raise NotImplementedError()

    def _write(
        self,
        step: adios2py.Step,
        variables: Mapping[str, Variable],
        attributes: Mapping[str, Any],
    ) -> None:
        for name, var in variables.items():
            step[name] = var
            dims = var.encoding.pop("prepend dimensions", "").split() + list(var.dims)
            step[name].attrs["dimensions"] = " ".join(dims)
            step[name].attrs["dtype"] = str(var.dtype)
            for attr_name, attr in var.attrs.items():
                step[name].attrs[attr_name] = attr

        for attr_name, attr in attributes.items():
            step.attrs[attr_name] = attr
