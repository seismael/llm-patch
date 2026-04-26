"""CLI command registry (Composite pattern).

The :class:`CommandRegistry` is a thin wrapper around ``click.Group`` that
adds the concept of *primary* vs *advanced/hidden* commands. Primary
commands appear in ``--help``; advanced commands are registered all the
same but are only listed when ``LLM_PATCH_ADVANCED=1`` is set in the
environment.

This keeps the user-facing surface friendly (init / compile / chat /
push / pull / doctor / version) while preserving the full power-user
surface (adapter / source / model / wiki / serve / hub) for those who
opt in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

import click

_ADVANCED_ENV_VAR: Final = "LLM_PATCH_ADVANCED"


def advanced_mode_enabled() -> bool:
    """Return True if the user has opted into advanced/hidden commands."""
    return os.environ.get(_ADVANCED_ENV_VAR, "").strip() not in ("", "0", "false", "False")


@dataclass
class _Entry:
    command: click.Command
    hidden: bool


@dataclass
class CommandRegistry:
    """Register click commands with a ``hidden`` flag.

    Hidden commands are still attached to the underlying ``click.Group``
    (so they remain reachable by name) but are filtered out of
    ``--help`` listings unless :func:`advanced_mode_enabled` is True.
    """

    group: click.Group
    _entries: dict[str, _Entry] = field(default_factory=dict)

    def register(
        self,
        command: click.Command,
        *,
        name: str | None = None,
        hidden: bool = False,
    ) -> None:
        """Attach *command* to the group under *name* (default: command.name)."""
        resolved = name or command.name
        if resolved is None:  # pragma: no cover - defensive
            raise ValueError("command must have a name")
        # Click respects the ``hidden`` attribute on Command for help listings.
        if hidden and not advanced_mode_enabled():
            command.hidden = True
        self.group.add_command(command, name=resolved)
        self._entries[resolved] = _Entry(command=command, hidden=hidden)

    def names(self, *, include_hidden: bool = False) -> list[str]:
        """Return registered command names (sorted)."""
        return sorted(
            name
            for name, entry in self._entries.items()
            if include_hidden or not entry.hidden or advanced_mode_enabled()
        )


__all__ = ["CommandRegistry", "advanced_mode_enabled"]
