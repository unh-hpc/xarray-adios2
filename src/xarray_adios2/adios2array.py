from __future__ import annotations

from typing import TYPE_CHECKING, Any

import adios2py
from numpy.typing import NDArray
from xarray.backends.common import BackendArray
from xarray.core import indexing

if TYPE_CHECKING:
    from .adios2store import Adios2Store


class Adios2Array(BackendArray):
    """Lazy evaluation of a variable stored in an adios2 file.

    This also takes care of slicing out the specific component of the data stored as 4-d array.
    """

    def __init__(
        self,
        variable_name: str,
        datastore: Adios2Store,
    ) -> None:
        self.variable_name = variable_name
        self.datastore = datastore
        array = self.get_array()
        self.shape = array.shape
        self.dtype = array.dtype

    def get_array(self, needs_lock: bool = True) -> adios2py.ArrayProxy:
        return self.datastore.acquire(needs_lock)[self.variable_name]

    def __getitem__(self, key: indexing.ExplicitIndexer) -> NDArray[Any]:
        return indexing.explicit_indexing_adapter(  # type: ignore[no-any-return]
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key) -> NDArray[Any]:  # type: ignore[no-untyped-def]
        with self.datastore.lock:
            return self.get_array(needs_lock=False)[key]
