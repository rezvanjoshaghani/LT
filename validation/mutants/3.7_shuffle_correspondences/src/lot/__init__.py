"""Learn or Transport? Geometry core and evaluation harness."""

import importlib

__all__ = [
    "correspondence",
    "encoders",
    "geometry",
    "render_replica",
    "transport",
    "visibility",
]


def __getattr__(name: str):
    """Import submodules on first attribute access.

    Importing them here instead would also import the module that is about to
    run under python -m, which executes that module twice and warns.
    """
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
