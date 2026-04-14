"""Weight generation strategies (Strategy Pattern)."""

import contextlib

with contextlib.suppress(ImportError, OSError):
    from llm_patch.generators.sakana_t2l import SakanaT2LGenerator

__all__ = ["SakanaT2LGenerator"]
