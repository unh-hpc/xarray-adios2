from __future__ import annotations

import os
from typing import Any, Protocol

import adios2py
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
        manager: FileManager | adios2py.File,
        mode: str | None = None,
        lock: Lock = ADIOS2_LOCK,
        autoclose: bool = False,
    ):
        if isinstance(manager, adios2py.File):
            mode = manager._mode
            manager = DummyFileManager(manager)  # type: ignore[no-untyped-call]

        self._manager = manager
        self._mode = mode
        self.lock = ensure_lock(lock)  # type: ignore[no-untyped-call]
        self.autoclose = autoclose

    @classmethod
    def open(
        cls,
        filename: str | os.PathLike[Any],
        mode: str = "rra",
        lock: Lock | None = None,
        autoclose: bool = False,
    ) -> Adios2Store:
        if lock is None:
            if mode in ("r", "rra"):
                lock = ADIOS2_LOCK
            else:
                lock = combine_locks([ADIOS2_LOCK, get_write_lock(filename)])  # type: ignore[no-untyped-call]

        manager = CachingFileManager(adios2py.File, filename, mode=mode)
        return cls(manager, mode=mode, lock=lock, autoclose=autoclose)

    def _acquire(self, needs_lock: bool = True) -> adios2py.Group:
        with self._manager.acquire_context(needs_lock) as group:  # type: ignore[no-untyped-call]
            ds = group
        assert isinstance(ds, adios2py.Group)
        return ds

    @property
    def ds(self) -> adios2py.Group:
        return self._acquire()
