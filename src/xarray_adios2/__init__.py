"""
Copyright (c) 2025 Kai Germaschewski. All rights reserved.

xarray-adios2: An xarray backend to read/write ADIOS2 files/streams.
"""

from __future__ import annotations

from ._version import version as __version__
from .adios2backend import Adios2BackendEntrypoint
from .adios2store import Adios2Store

__all__ = [
    "Adios2BackendEntrypoint",
    "Adios2Store",
    "__version__",
]
