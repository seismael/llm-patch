"""Preflight checks for hardware-sensitive commands.

Used by ``llm-patch compile`` / ``watch`` / ``chat`` to give the user a
fast, friendly error long before the heavy ML stack is loaded. Pure
stdlib + lazy ``torch`` probe — does NOT import torch unless explicitly
asked.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreflightReport:
    """Summary of detected hardware capabilities.

    Attributes:
        cuda_available: Whether ``torch.cuda.is_available()`` returns True.
        device_count: Number of CUDA devices (0 if unavailable).
        device_name: First device name, or ``None``.
        total_vram_bytes: Total VRAM on device 0, or 0.
        free_vram_bytes: Free VRAM on device 0, or 0.
        torch_imported: Whether the probe successfully imported torch.
    """

    cuda_available: bool
    device_count: int
    device_name: str | None
    total_vram_bytes: int
    free_vram_bytes: int
    torch_imported: bool

    def has_min_vram(self, gb: float) -> bool:
        """Return True if free VRAM ≥ *gb* gibibytes."""
        return self.free_vram_bytes >= int(gb * (1024**3))

    def human_summary(self) -> str:
        """Human-readable single-line summary."""
        if not self.torch_imported:
            return "torch not installed; CPU-only execution."
        if not self.cuda_available:
            return "CUDA unavailable; CPU-only execution."
        gib = self.free_vram_bytes / (1024**3)
        return (
            f"{self.device_name} | "
            f"{self.device_count} device(s) | "
            f"{gib:.1f} GiB free VRAM"
        )


def probe(*, import_torch: bool = True) -> PreflightReport:
    """Probe local hardware capabilities.

    Args:
        import_torch: When ``False``, skip the import (returns a
            CPU-only stub). Useful for ``llm-patch --help`` style
            fast paths.
    """
    if not import_torch:
        return PreflightReport(
            cuda_available=False,
            device_count=0,
            device_name=None,
            total_vram_bytes=0,
            free_vram_bytes=0,
            torch_imported=False,
        )

    try:
        import torch
    except ImportError:
        return PreflightReport(
            cuda_available=False,
            device_count=0,
            device_name=None,
            total_vram_bytes=0,
            free_vram_bytes=0,
            torch_imported=False,
        )

    if not torch.cuda.is_available():
        return PreflightReport(
            cuda_available=False,
            device_count=0,
            device_name=None,
            total_vram_bytes=0,
            free_vram_bytes=0,
            torch_imported=True,
        )

    count = int(torch.cuda.device_count())
    name = str(torch.cuda.get_device_name(0)) if count else None
    free, total = (0, 0)
    if count:
        try:
            free, total = torch.cuda.mem_get_info(0)
        except (RuntimeError, AttributeError):
            free, total = (0, 0)
    return PreflightReport(
        cuda_available=True,
        device_count=count,
        device_name=name,
        total_vram_bytes=int(total),
        free_vram_bytes=int(free),
        torch_imported=True,
    )


__all__ = ["PreflightReport", "probe"]
