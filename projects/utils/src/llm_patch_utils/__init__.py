"""llm_patch_utils — cross-project utilities for the llm-patch monorepo.

This package is intentionally minimal. It hosts only primitives that are
genuinely needed by two or more projects in the workspace (e.g., logging,
config helpers, telemetry hooks, common error types).

**Policy**: stdlib-only runtime dependencies. Adding any third-party
dependency requires an ADR under ``docs/adr/``. **Never** depend on
``llm_patch`` or any use-case package — dependencies flow downward only
(see ``SPEC.md``: Dependency Direction).
"""

__version__ = "0.2.0"

from llm_patch_utils.errors import (
    AdapterNotFoundError,
    CapacityExceededError,
    ChecksumMismatchError,
    ConfigurationError,
    DependencyError,
    IncompatibleBaseModelError,
    IntegrationError,
    LlmPatchError,
    RegistryUnavailableError,
    ResourceNotFoundError,
)

__all__ = [
    "AdapterNotFoundError",
    "CapacityExceededError",
    "ChecksumMismatchError",
    "ConfigurationError",
    "DependencyError",
    "IncompatibleBaseModelError",
    "IntegrationError",
    "LlmPatchError",
    "RegistryUnavailableError",
    "ResourceNotFoundError",
    "__version__",
]
