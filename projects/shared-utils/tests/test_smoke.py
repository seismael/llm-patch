"""Smoke tests for ``llm_patch_shared`` package metadata and exports."""

from __future__ import annotations

import llm_patch_shared
from llm_patch_shared import (
    ConfigurationError,
    DependencyError,
    IntegrationError,
    LlmPatchError,
    ResourceNotFoundError,
)


def test_version_exposed() -> None:
    assert isinstance(llm_patch_shared.__version__, str)
    assert llm_patch_shared.__version__.count(".") >= 2


def test_error_hierarchy() -> None:
    assert issubclass(ConfigurationError, LlmPatchError)
    assert issubclass(DependencyError, IntegrationError)
    assert issubclass(IntegrationError, LlmPatchError)
    assert issubclass(ResourceNotFoundError, IntegrationError)
