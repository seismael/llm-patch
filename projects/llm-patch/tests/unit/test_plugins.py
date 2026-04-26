"""Tests for :mod:`llm_patch.core.plugins`."""

from __future__ import annotations

import pytest

from llm_patch.core.plugins import (
    ENTRY_POINT_GROUP,
    ENV_VAR_PREFIX,
    PluginKind,
    PluginLoader,
    PluginSpec,
)


def test_plugin_spec_parse_round_trip() -> None:
    spec = PluginSpec.parse("my_pkg.module:factory", origin="test")
    assert spec.module == "my_pkg.module"
    assert spec.attribute == "factory"
    assert spec.origin == "test"


@pytest.mark.parametrize("bad", ["", "no_colon", ":missing_module", "missing_attr:"])
def test_plugin_spec_parse_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        PluginSpec.parse(bad, origin="test")


def test_plugin_spec_resolve_imports_attribute() -> None:
    spec = PluginSpec.parse("json:loads", origin="test")
    fn = spec.resolve()
    assert callable(fn)
    assert fn('{"a": 1}') == {"a": 1}


def test_plugin_spec_resolve_missing_attribute_raises_import_error() -> None:
    spec = PluginSpec.parse("json:does_not_exist_xyz", origin="test")
    with pytest.raises(ImportError):
        spec.resolve()


def test_plugin_loader_env_resolution() -> None:
    env = {f"{ENV_VAR_PREFIX}REGISTRY": "json:loads"}
    loader = PluginLoader(env=env)
    fn = loader.resolve(PluginKind.REGISTRY)
    assert fn is not None
    assert fn('{"x": 2}') == {"x": 2}


def test_plugin_loader_env_returns_none_when_unset() -> None:
    loader = PluginLoader(env={})
    assert loader.resolve(PluginKind.REGISTRY) is None


def test_plugin_loader_env_spec_metadata() -> None:
    env = {f"{ENV_VAR_PREFIX}SOURCE": "json:loads"}
    spec = PluginLoader(env=env).env_spec(PluginKind.SOURCE)
    assert spec is not None
    assert spec.module == "json"
    assert spec.attribute == "loads"
    assert spec.origin.startswith("env:")


def test_plugin_loader_entry_point_specs_returns_list() -> None:
    # No assumptions about which entry points are installed in CI; just
    # validate that the discovery path executes and returns a list.
    specs = PluginLoader().entry_point_specs()
    assert isinstance(specs, list)


def test_entry_point_group_constant() -> None:
    assert ENTRY_POINT_GROUP == "llm_patch.plugins"


def test_plugin_kind_values() -> None:
    assert {k.value for k in PluginKind} == {
        "SOURCE",
        "GENERATOR",
        "LOADER",
        "RUNTIME",
        "REGISTRY",
        "CACHE",
        "CONTROLLER",
    }


def test_public_reexports() -> None:
    import llm_patch

    assert llm_patch.__version__ == "1.0.0rc1"
    assert llm_patch.PluginLoader is PluginLoader
    assert llm_patch.PluginKind is PluginKind
    assert "PluginLoader" in llm_patch.__all__
    assert "PluginKind" in llm_patch.__all__
    assert "PluginSpec" in llm_patch.__all__
