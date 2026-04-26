# Quickstart — `llm-patch` in 5 minutes

> Goal: turn a folder of notes into a LoRA adapter, attach it to a base
> model, and chat with the result. Pure CLI, no Python required.

If you already know the engine and want the framework story, jump
straight to [ARCHITECTURE.md](ARCHITECTURE.md). For the full reference,
see [USAGE.md](USAGE.md).

---

## 1. Install

```pwsh
pip install "llm-patch[cli]"
```

The `[cli]` extra pulls in `click`. The base model + PEFT toolchain are
installed lazily by the commands that actually need them — `--help`
itself never imports torch.

> **Verify your environment first.** `llm-patch doctor` prints Python,
> torch, CUDA, optional-extra status, and registry config in one shot.

```pwsh
llm-patch doctor
```

## 2. Scaffold a project

```pwsh
llm-patch init
```

`init` is interactive. It asks:

- project name
- source directory (defaults to `./docs`)
- base model id (defaults to `google/gemma-2-2b-it`)
- output directory (defaults to `./adapters`)

…and writes a `.llm-patch.toml` in the current directory. Use
`--non-interactive` plus flags to script the same thing in CI.

## 3. Compile a folder into an adapter

Drop a few markdown notes into the source directory, then:

```pwsh
llm-patch compile ./docs --output ./adapters
```

This runs the
[`Ingest → Compile`](ARCHITECTURE.md#system-overview) half of the
pipeline: each document becomes a `~2–5 MB` LoRA adapter saved as
`safetensors`, with a v2 manifest next to it.

## 4. Chat with the patched model

```pwsh
llm-patch chat --base google/gemma-2-2b-it --adapter ./adapters/my-notes
```

The base model is loaded once; the adapter is attached on top; you get
a REPL whose responses are conditioned on your notes — without any of
those notes consuming context-window tokens.

## 5. Share what you built (optional)

If you operate (or join) an adapter hub, point `llm-patch` at it and
publish:

```pwsh
$Env:LLM_PATCH_PLUGIN_REGISTRY = "my_org_registry:build_registry"
llm-patch push ./adapters/my-notes --target hub://acme/my-notes:0.1.0
llm-patch hub search "notes"
llm-patch pull hub://acme/my-notes:0.1.0
```

The hub protocol is documented in [REGISTRY_PROTOCOL.md](REGISTRY_PROTOCOL.md).
The end-to-end agent story (search → pull → hot-swap) lives in
[AGENTIC_AI_INTEGRATION.md](AGENTIC_AI_INTEGRATION.md).

---

## Where to next

| Goal | Read |
|---|---|
| Full CLI reference | [USAGE.md §CLI](USAGE.md#cli-reference) |
| Plug a new source / generator / registry into the CLI | [EXTENDING.md](EXTENDING.md) |
| Understand the engine internals | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Distributed adapter market | [AGENTIC_AI_INTEGRATION.md](AGENTIC_AI_INTEGRATION.md) |
| Contribute / ask questions | [COMMUNITY.md](COMMUNITY.md) |

---

## Troubleshooting one-liners

| Symptom | Run |
|---|---|
| Anything broken | `llm-patch doctor --json` |
| `--help` slow | Open a bug — `--help` should be torch-free |
| `push`/`pull` fails with `RegistryUnavailableError` | Set `LLM_PATCH_PLUGIN_REGISTRY="module:factory"` ([details](EXTENDING.md#registries)) |
| Out of VRAM during `chat` | `llm-patch doctor` reports min-VRAM fit per base model |
