"""Plugin discovery for llm-patch.

Resolves user-supplied plugins (sources, generators, registry clients,
runtimes, etc.) using two complementary mechanisms:

1. **Environment variables** — ``LLM_PATCH_PLUGIN_<KIND>=module:factory``
   for ad-hoc, single-user setups.
2. **Entry points** — packages can advertise plugins via the
   ``llm_patch.plugins`` entry-point group in their ``pyproject.toml``::

       [project.entry-points."llm_patch.plugins"]
       my_registry = "my_pkg.registry:build"

The loader is intentionally minimal: it returns the *factory callable*,
not an instantiated object. Callers decide when to invoke it (so that
plugin construction stays under the user's control and can fail loudly
with a meaningful traceback).

This module imports nothing heavy — it stays in the lazy-CLI fast path.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Final

ENTRY_POINT_GROUP: Final = "llm_patch.plugins"
ENV_VAR_PREFIX: Final = "LLM_PATCH_PLUGIN_"


class PluginKind(str, Enum):
    """Supported plugin kinds.

    String values match the suffix used in the corresponding environment
    variable (uppercased). For example, ``REGISTRY`` resolves to
    ``LLM_PATCH_PLUGIN_REGISTRY``.
    """

    SOURCE = "SOURCE"
    GENERATOR = "GENERATOR"
    LOADER = "LOADER"
    RUNTIME = "RUNTIME"
    REGISTRY = "REGISTRY"
    CACHE = "CACHE"
    CONTROLLER = "CONTROLLER"


@dataclass(frozen=True)
class PluginSpec:
    """Resolved plugin reference: ``module:attribute`` plus origin tag."""

    module: str
    attribute: str
    origin: str  # "env" | "entry_point"
    name: str | None = None  # only set for entry-point plugins

    @classmethod
    def parse(cls, spec: str, *, origin: str, name: str | None = None) -> "PluginSpec":
        """Parse a ``module:attribute`` string into a :class:`PluginSpec`."""
        if ":" not in spec:
            raise ValueError(
                f"Invalid plugin spec {spec!r}: expected 'module:attribute' format."
            )
        module, _, attribute = spec.partition(":")
        if not module or not attribute:
            raise ValueError(
                f"Invalid plugin spec {spec!r}: both module and attribute are required."
            )
        return cls(module=module.strip(), attribute=attribute.strip(), origin=origin, name=name)

    def resolve(self) -> Callable[..., Any]:
        """Import the module and return the attribute (typically a factory)."""
        mod = importlib.import_module(self.module)
        try:
            return getattr(mod, self.attribute)
        except AttributeError as exc:
            raise ImportError(
                f"Plugin {self.module}:{self.attribute} not found "
                f"(origin={self.origin})."
            ) from exc


class PluginLoader:
    """Resolve plugins from environment variables and entry points.

    Single-responsibility: discovery only. Construction of plugin
    instances (and any error reporting) is left to the caller.
    """

    def __init__(
        self,
        *,
        env: dict[str, str] | None = None,
        entry_point_group: str = ENTRY_POINT_GROUP,
    ) -> None:
        self._env = dict(env) if env is not None else dict(os.environ)
        self._entry_point_group = entry_point_group

    # ── Environment-variable discovery ───────────────────────────────
    def env_spec(self, kind: PluginKind) -> PluginSpec | None:
        """Return the plugin spec from ``LLM_PATCH_PLUGIN_<KIND>`` if set."""
        var = f"{ENV_VAR_PREFIX}{kind.value}"
        raw = self._env.get(var, "").strip()
        if not raw:
            return None
        return PluginSpec.parse(raw, origin=f"env:{var}")

    # ── Entry-point discovery ────────────────────────────────────────
    def entry_point_specs(self) -> list[PluginSpec]:
        """Return all entry-point specs in the configured group."""
        try:
            eps = entry_points(group=self._entry_point_group)
        except TypeError:  # pragma: no cover - py<3.10 fallback path
            eps = entry_points().get(self._entry_point_group, [])
        return [self._spec_from_entry_point(ep) for ep in eps]

    @staticmethod
    def _spec_from_entry_point(ep: EntryPoint) -> PluginSpec:
        # ``ep.value`` is the canonical "module:attribute[ extras]".
        target = ep.value.split()[0]
        return PluginSpec.parse(target, origin="entry_point", name=ep.name)

    # ── Combined resolution ──────────────────────────────────────────
    def resolve(self, kind: PluginKind) -> Callable[..., Any] | None:
        """Resolve a plugin of *kind*, env-var winning over entry points.

        Returns ``None`` if no plugin is registered for this kind.
        """
        spec = self.env_spec(kind)
        if spec is not None:
            return spec.resolve()
        # Entry points are not currently keyed by kind. Future versions
        # may introduce a structured mapping; for now, return None and
        # let callers iterate over :meth:`entry_point_specs` directly.
        return None


__all__ = [
    "ENTRY_POINT_GROUP",
    "ENV_VAR_PREFIX",
    "PluginKind",
    "PluginLoader",
    "PluginSpec",
]
