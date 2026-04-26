# Limitations and Non-Goals

This document records what `llm-patch` **does not** do today, and which
items are explicitly outside the v1.x scope. Reading it first will save
you from chasing features that aren't there.

## Adapter Compilation

- **Hypernetwork-only**: adapter weights are produced by a Text-to-LoRA
  (T2L) hypernetwork. There is no in-process gradient-based fine-tuning
  loop. If you need supervised fine-tuning on labeled data, use
  [PEFT](https://github.com/huggingface/peft) directly.
- **No distributed training**: a single-process `compile` runs against a
  single device (`--device cpu` / `cuda` / `cuda:N`). DDP, FSDP, and
  pipeline parallelism are out of scope for v1.
- **CPU compile is best-effort**: T2L was trained on GPUs; CPU runs are
  numerically equivalent but may be slow on large adapters.

## Runtime Inference

- **No quantized generation**: bitsandbytes/4-bit/8-bit base models are
  not supported by the default `HFModelProvider`. Wire your own
  `IModelProvider` if you need them.
- **No streaming output yet**: `llm-patch chat` and `model generate`
  return complete responses; token streaming is planned for a 1.x minor
  release.
- **No multi-adapter routing**: when several adapters are attached, the
  runtime activates them additively. Per-prompt routing
  (mixture-of-adapters) is not implemented.

## Registry & Distribution

- **No bundled registry client**: the engine ships zero concrete
  `IAdapterRegistryClient` implementations. Operators wire one via
  `LLM_PATCH_PLUGIN_REGISTRY="module:factory"` — see
  [REGISTRY_PROTOCOL.md](REGISTRY_PROTOCOL.md). Reference clients live
  outside the engine repository (community-maintained).
- **No automatic signing or attestation**: manifest checksums are
  SHA-256 over the adapter blob; signing is out of scope for v1. If you
  need supply-chain integrity, sign manifests externally and verify on
  pull.
- **No ACL/quota enforcement in `serve`**: the bundled HTTP server is a
  reference implementation. Production deployments should sit behind a
  reverse proxy that handles authn/z, rate limiting, and TLS.

## Platforms

- **Linux is the primary platform**. The full test matrix runs on Linux
  + Python 3.11 / 3.12.
- **Windows and macOS are best-effort**: CLI verbs and pure-Python
  features are tested locally but not in CI. File-watcher behavior
  varies by OS (notably, `llm-patch watch` Ctrl-C handling on Windows).
- **Python 3.10 is not supported**: the codebase uses 3.11+ syntax and
  stdlib `tomllib`.

## Security

- **No sandboxing of plugin code**: the registry plugin loader executes
  whatever module is named in `LLM_PATCH_PLUGIN_REGISTRY`. Treat it the
  same way you treat `pip install` — only point it at modules you trust.
- **No PII redaction**: source documents flow into adapter weights
  verbatim (subject to the hypernetwork's compression). Do not compile
  documents you wouldn't be willing to share with downstream adapter
  consumers.

## Out of Scope (Indefinitely)

The following are **non-goals** — pull requests adding them will be
declined:

- Bundling a hosted adapter registry inside the engine.
- A web UI for browsing adapters (the CLI + an external registry's UI
  are sufficient).
- A general-purpose RAG framework (compose `llm-patch` with your
  preferred RAG stack instead — they solve different problems).
- A fine-tuning trainer (use PEFT, axolotl, etc.).

If a limitation here blocks you, please open an issue describing the
use-case before submitting changes — most can be addressed by a plugin
or a thin downstream wrapper without changing the engine.
