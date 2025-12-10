# modeling/factory.py
from typing import Any, Dict, Optional, Type
import inspect

_PROJECTORS: Dict[str, Type[Any]] = {}
_FUSERS: Dict[str, Type[Any]] = {}

def _check_and_register(registry: Dict[str, Type[Any]], name: str, cls: Type[Any], kind: str):
    if name in registry and registry[name] is not cls:
        raise ValueError(
            f"{kind} name '{name}' already registered with {registry[name].__module__}.{registry[name].__name__}; "
            f"refusing to overwrite with {cls.__module__}.{cls.__name__}."
        )
    registry[name] = cls

def register_projector(*names: str):
    """
    Usage:
        @register_projector("residual_ffn", "residual-ffn", "ResidualFFN")
        class ResidualFFNProjector(...): ...
    """
    if not names:
        raise ValueError("register_projector requires at least one name/alias.")
    def wrap(cls):
        for n in names:
            _check_and_register(_PROJECTORS, n, cls, "Projector")
        return cls
    return wrap

def register_fuser(*names: str):
    if not names:
        raise ValueError("register_fuser requires at least one name/alias.")
    def wrap(cls):
        for n in names:
            _check_and_register(_FUSERS, n, cls, "Fuser")
        return cls
    return wrap

def _build(registry: Dict[str, Type[Any]], kind: str, name: str, **kwargs):
    cls = registry.get(name)
    if cls is None:
        available = ", ".join(sorted(registry.keys())) or "(none registered)"
        raise KeyError(
            f"{kind} '{name}' not found. Available {kind.lower()}s: {available}. "
            "If this should exist, ensure the module that registers it is imported before calling build."
        )
    # Helpful kwargs check
    sig = inspect.signature(cls.__init__)
    try:
        sig.bind_partial(None, **kwargs)  # 'self' placeholder
    except TypeError as e:
        raise TypeError(
            f"{kind} '{name}' received incompatible kwargs {kwargs}. "
            f"Constructor signature: {cls.__module__}.{cls.__name__}{sig}"
        ) from e
    return cls(**kwargs)

def build_projector(name: str, **kwargs):
    return _build(_PROJECTORS, "Projector", name, **kwargs)

def build_fuser(name: str, **kwargs):
    return _build(_FUSERS, "Fuser", name, **kwargs)

def list_projectors() -> Dict[str, str]:
    return {k: f"{v.__module__}.{v.__name__}" for k, v in _PROJECTORS.items()}

def list_fusers() -> Dict[str, str]:
    return {k: f"{v.__module__}.{v.__name__}" for k, v in _FUSERS.items()}
