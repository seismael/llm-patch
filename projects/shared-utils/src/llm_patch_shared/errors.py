"""Common error hierarchy for the llm-patch monorepo.

All projects in the workspace should derive their domain exceptions from
:class:`LlmPatchError` so that downstream callers can catch a single
common base when they don't care about the specific subtype.
"""

from __future__ import annotations


class LlmPatchError(Exception):
    """Root of the llm-patch error hierarchy."""


class ConfigurationError(LlmPatchError):
    """Raised when configuration is missing or invalid."""


class IntegrationError(LlmPatchError):
    """Raised when an external integration (model, storage, API) fails."""


class DependencyError(IntegrationError):
    """Raised when an optional runtime dependency is unavailable."""


class ResourceNotFoundError(IntegrationError):
    """Raised when a requested runtime resource cannot be found."""
