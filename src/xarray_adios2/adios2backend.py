from __future__ import annotations

import os
import pathlib
from collections.abc import Iterable
from typing import Any

from typing_extensions import override
from xarray.backends.common import AbstractDataStore, BackendEntrypoint, _normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.types import ReadBuffer

from .adios2store import Adios2Store


class Adios2BackendEntrypoint(BackendEntrypoint):
    """Entrypoint that lets xarray recognize and read ADIOS2 output."""

    open_dataset_parameters = ("filename_or_obj", "drop_variables")
    # url =
    available = True

    @override
    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
    ) -> bool:
        if isinstance(filename_or_obj, str | os.PathLike):
            ext = pathlib.Path(filename_or_obj).suffix
            return ext in {".bp"}

        return False

    @override
    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
        *,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        decode_coords: bool = True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime: bool | None = None,
        decode_timedelta: bool | None = None,
    ) -> Dataset:
        filename = _normalize_path(filename_or_obj)

        assert isinstance(filename, str | os.PathLike)
        store = Adios2Store.open(filename, mode="rra")

        store_entrypoint = StoreBackendEntrypoint()

        return store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

    @override
    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer[Any] | AbstractDataStore,
        **kwargs: Any,
    ) -> DataTree:
        raise NotImplementedError()
