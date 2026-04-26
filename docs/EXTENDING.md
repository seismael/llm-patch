# Extending llm-patch — author and ship a plugin

`llm-patch` is built around a small, stable set of ABCs. Anything you
can imagine as a *source*, a *weight generator*, a *registry transport*,
or a *runtime controller* is a plugin. This guide shows how to author
one and ship it as an installable package.

> **Public API only.** Use-case projects and plugins must consume only
> symbols re-exported from `llm_patch` (the top-level package). See
> [docs/adr/0003-public-api-policy.md](adr/0003-public-api-policy.md).

---

## 1. Pick the ABC

| Plugin kind  | ABC                              | Use it for…                                          |
|---           |---                               |---                                                   |
| `SOURCE`     | `IDataSource`                    | Wikis, PDFs, HTTP APIs, custom note formats.         |
| `GENERATOR`  | `IWeightGenerator`               | Alternative hypernetworks or distillation pipelines. |
| `LOADER`     | `IAdapterLoader`                 | Custom adapter merge / quantization strategies.      |
| `RUNTIME`    | `IAgentRuntime`                  | Non-PEFT serving paths.                              |
| `REGISTRY`   | `IAdapterRegistryClient`         | Hub transports — HTTP, S3, OCI, local-FS.            |
| `CACHE`      | `IAdapterCache`                  | LRU replacements, distributed caches.                |
| `CONTROLLER` | `IRuntimeAdapterController`      | Hot-swap policies and concurrency models.            |

```python
from llm_patch import IAdapterRegistryClient
```

---

## 2. Implement the ABC

A registry client only needs three methods:

```python
# my_registry/__init__.py
from llm_patch import AdapterRef, IAdapterRegistryClient

class HttpRegistry(IAdapterRegistryClient):
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def search(self, query: str) -> list[AdapterRef]:
        ...
    def fetch(self, ref: AdapterRef, *, dest) -> None:
        ...
    def publish(self, manifest_path) -> AdapterRef:
        ...

def build_registry() -> HttpRegistry:
    """Factory used by llm-patch plugin discovery."""
    return HttpRegistry(base_url="https://hub.example.com")
```

Tips:

* Keep the constructor side-effect-free.
* Raise the engine's typed errors (`from llm_patch_shared.errors import …`)
  so the CLI can render them consistently.

---

## 3. Wire it up — two ways

### A) Environment variable (single user, no packaging)

```pwsh
$Env:LLM_PATCH_PLUGIN_REGISTRY = "my_registry:build_registry"
llm-patch doctor      # confirms the plugin is configured
llm-patch push ./adapters/my-notes
```

### B) Entry point (distributable package)

In your plugin's `pyproject.toml`:

```toml
[project.entry-points."llm_patch.plugins"]
http_registry = "my_registry:build_registry"
```

Install it (`pip install my-registry`) and the loader picks it up
automatically. Env vars still take precedence for per-process overrides.

---

## 4. Confirm the wiring with `doctor`

```pwsh
llm-patch doctor
```

The `Registry plugin:` line shows the resolved spec, or a friendly hint
if no plugin is configured.

---

## 5. Test your plugin

The engine ships its tests with public-API contracts you can reuse:

```python
from llm_patch import AdapterRef, IAdapterRegistryClient

def test_http_registry_satisfies_abc() -> None:
    from my_registry import HttpRegistry
    assert issubclass(HttpRegistry, IAdapterRegistryClient)
```

For end-to-end verification, run the CLI with your plugin loaded:

```pwsh
$Env:LLM_PATCH_PLUGIN_REGISTRY = "my_registry:build_registry"
llm-patch hub search "react"
```

---

## 6. Publish

* Add a `README.md` describing the plugin kind and any auth requirements.
* Tag a release and `pip install` from a fresh venv to validate.
* Open a PR against `docs/COMMUNITY.md` to add your plugin to the
  community gallery.

---

## See also

* [docs/REGISTRY_PROTOCOL.md](REGISTRY_PROTOCOL.md) — wire format for
  registry-client plugins.
* [docs/SERVER_ARCHITECTURE.md](SERVER_ARCHITECTURE.md) — runtime /
  controller plugin contract.
* [docs/adr/0008-plugin-discovery.md](adr/0008-plugin-discovery.md) —
  decision record for this mechanism.
