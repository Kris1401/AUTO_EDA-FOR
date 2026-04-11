"""Data Chat branches package.

Exports branch modules so pages can import deterministically.

Keep imports light (module-level only). Each branch should expose:
- render(df, ctx) -> dict
"""

from . import distribution
from . import composition_static
from . import composition_over_time
from . import _chart_bundle

__all__ = [
    "distribution",
    "composition_static",
    "composition_over_time",
    "_chart_bundle",
]
