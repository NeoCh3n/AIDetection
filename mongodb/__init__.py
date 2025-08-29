"""MongoDB utilities package.

This package exposes commonly used submodules at the package level
so tools like Pylance can resolve names listed in ``__all__``.

Python 3.6.8 compatible.
"""

# Re-export submodules so names exist on the package object
from . import insert_DB as insert_DB  # noqa: F401
from . import query_DB as query_DB  # noqa: F401
from . import delete_DB as delete_DB  # noqa: F401
from . import mongodb_connection as mongodb_connection  # noqa: F401

__all__ = [
    "insert_DB",
    "query_DB",
    "delete_DB",
    "mongodb_connection",
]
