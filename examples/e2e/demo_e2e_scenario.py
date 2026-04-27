#!/usr/bin/env python
"""End-to-end scenario: Wiki changes -> LoRA pipeline -> model answers improve.

This is the definitive demo of llm-patch. It runs a scripted scenario that
proves the full pipeline works step by step:

  Step 1: Empty wiki -- model gives generic answers (no domain knowledge)
  Step 2: Add Transformer paper -> adapter generated -> answers improve
  Step 3: Add LoRA paper -> second adapter -> answers improve further
  Step 4: Update Transformer page with more detail -> adapter refreshes
  Step 5: Start live watcher -> drop a new file -> auto-detected and processed

Each step asks the SAME questions and prints a side-by-side comparison so you
can see the knowledge accumulating.

No GPU required. Runs entirely on mock components that use real wiki content.

Usage:
    python demo_e2e_scenario.py
    python demo_e2e_scenario.py --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import textwrap
import time
from pathlib import Path
from typing import Any

from llm_patch.core.config import WatcherConfig
from llm_patch.core.interfaces import IAdapterRepository, IWeightGenerator
from llm_patch.core.models import AdapterManifest, DocumentContext
from llm_patch.orchestrator import KnowledgeFusionOrchestrator
from llm_patch.sources.wiki_source import WikiKnowledgeSource

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Components — Content-aware generator + model that uses adapter knowledge
# ═══════════════════════════════════════════════════════════════════════════


class ContentAwareGenerator(IWeightGenerator):
    """Weight generator that encodes actual document content into the adapter.

    Instead of generating real tensor weights (which would need torch + a
    hypernetwork checkpoint), this stores the document text and metadata as
    the adapter payload.  The KnowledgeEnhancedModel can then query this
    payload to produce informed answers.
    """

    def __init__(self, *, rank: int = 8) -> None:
        self._rank = rank

    def generate(self, context: DocumentContext) -> dict[str, Any]:
        return {
            "__doc_id__": context.document_id,
            "__content__": context.content,
            "__metadata__": dict(context.metadata),
            "__content_hash__": hashlib.sha256(context.content.encode()).hexdigest()[:12],
        }

    def get_peft_config(self) -> dict[str, Any]:
        return {
            "r": self._rank,
            "target_modules": ["q_proj", "v_proj"],
            "lora_alpha": self._rank * 2,
            "peft_type": "LORA",
        }


class InMemoryAdapterRepository(IAdapterRepository):
    """Stores adapters in memory with full content payloads."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[dict[str, Any], Any]] = {}

    def save(self, adapter_id: str, weights: dict[str, Any], peft_config: Any) -> AdapterManifest:
        self._store[adapter_id] = (weights, peft_config)
        rank = peft_config.get("r", 8) if isinstance(peft_config, dict) else 8
        modules = peft_config.get("target_modules", []) if isinstance(peft_config, dict) else []
        return AdapterManifest(
            adapter_id=adapter_id,
            rank=rank,
            target_modules=modules,
            storage_uri=f"memory://{adapter_id}",
        )

    def load(self, adapter_id: str) -> dict[str, Any]:
        return self._store[adapter_id][0]

    def exists(self, adapter_id: str) -> bool:
        return adapter_id in self._store

    def list_adapters(self) -> list[AdapterManifest]:
        return [
            AdapterManifest(
                adapter_id=aid,
                rank=8,
                target_modules=["q_proj", "v_proj"],
                storage_uri=f"memory://{aid}",
            )
            for aid in self._store
        ]

    def delete(self, adapter_id: str) -> None:
        self._store.pop(adapter_id, None)

    @property
    def adapter_count(self) -> int:
        return len(self._store)

    def get_all_content(self) -> dict[str, str]:
        """Return {adapter_id: content} for all stored adapters."""
        return {aid: weights.get("__content__", "") for aid, (weights, _) in self._store.items()}

    def get_content_hash(self, adapter_id: str) -> str:
        """Return content hash for a specific adapter."""
        if adapter_id in self._store:
            return self._store[adapter_id][0].get("__content_hash__", "n/a")
        return "n/a"


class KnowledgeEnhancedModel:
    """Simulated LLM that answers questions using adapter knowledge.

    Without adapters loaded, it returns only generic answers.
    With adapters, it searches the ingested wiki content and synthesizes
    an informed answer -- demonstrating that the pipeline genuinely
    transfers knowledge from wiki pages into the model.
    """

    def __init__(self, repository: InMemoryAdapterRepository) -> None:
        self._repo = repository

    def answer(self, question: str) -> str:
        """Answer a question, enhanced by any loaded adapter knowledge."""
        all_content = self._repo.get_all_content()

        if not all_content:
            return self._generic_answer(question)

        # Search adapter content for relevant passages
        relevant = self._find_relevant_passages(question, all_content)

        if not relevant:
            return self._generic_answer(question)

        return self._build_answer(question, relevant)

    def _generic_answer(self, question: str) -> str:
        return (
            "[BASE MODEL -- no adapters loaded]\n"
            "I don't have specific knowledge about this topic. "
            "My training data contains general information, but I cannot "
            "provide detailed or accurate answers about specialized research."
        )

    def _find_relevant_passages(
        self, question: str, content_map: dict[str, str]
    ) -> list[tuple[str, str, float]]:
        """Find passages relevant to the question using keyword matching."""
        q_lower = question.lower()
        q_words = set(re.findall(r"[a-z]{3,}", q_lower))

        results: list[tuple[str, str, float]] = []

        for adapter_id, content in content_map.items():
            if not content.strip():
                continue

            content_lower = content.lower()
            matched_words = sum(1 for w in q_words if w in content_lower)
            if matched_words == 0:
                continue

            score = matched_words / max(len(q_words), 1)

            # Extract the best matching paragraph
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            best_para = ""
            best_para_score = 0.0
            for para in paragraphs:
                para_lower = para.lower()
                para_matches = sum(1 for w in q_words if w in para_lower)
                para_score = para_matches / max(len(q_words), 1)
                if para_score > best_para_score:
                    best_para_score = para_score
                    best_para = para

            if best_para:
                results.append((adapter_id, best_para, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:3]

    def _build_answer(self, question: str, passages: list[tuple[str, str, float]]) -> str:
        adapter_ids = list(dict.fromkeys(p[0] for p in passages))
        n_adapters = len(adapter_ids)

        lines = [
            f"[ENHANCED MODEL -- {n_adapters} adapter(s) active: {', '.join(adapter_ids)}]",
            "",
            "Based on my specialized knowledge:",
            "",
        ]

        for adapter_id, passage, _score in passages:
            clean = re.sub(r"^#+\s+.*$", "", passage, flags=re.MULTILINE).strip()
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
            clean = re.sub(r"\$[^$]+\$", "[formula]", clean)
            clean = re.sub(r"\$\$[^$]+\$\$", "[equation]", clean)

            if clean:
                wrapped = textwrap.fill(clean, width=72)
                lines.append(f"  From {adapter_id}:")
                lines.append(f"    {wrapped}")
                lines.append("")

        return "\n".join(lines).rstrip()


# ═══════════════════════════════════════════════════════════════════════════
# Wiki content for each step
# ═══════════════════════════════════════════════════════════════════════════

QUESTIONS = [
    "What is the Transformer architecture and how does it work?",
    "What is LoRA and how does it reduce fine-tuning cost?",
    "How does self-attention compute the output?",
]

WIKI_TRANSFORMER_V1 = """\
---
title: Attention Is All You Need
authors: Vaswani et al.
year: 2017
tags: transformer, attention
---

# Attention Is All You Need

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks. We propose the [[Transformer]], a new architecture
based solely on [[Self-Attention]] mechanisms.

## Key Ideas

- **Self-Attention Mechanism**: Scaled dot-product attention computes compatibility
  between queries and keys, weighting values accordingly.
- **Multi-Head Attention**: Allows the model to attend to information from different
  representation sub-spaces at different positions.
- **Positional Encoding**: Added to give the model position information since there
  is no recurrence or convolution.

## Results

The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German translation,
improving over existing best results by over 2 BLEU.
"""

WIKI_TRANSFORMER_V2_UPDATE = """\
---
title: Attention Is All You Need
authors: Vaswani et al.
year: 2017
tags: transformer, attention
---

# Attention Is All You Need

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks. We propose the [[Transformer]], a new architecture
based solely on [[Self-Attention]] mechanisms.

## Key Ideas

- **Self-Attention Mechanism**: Scaled dot-product attention computes compatibility
  between queries and keys, weighting values accordingly.
- **Multi-Head Attention**: Allows the model to attend to information from different
  representation sub-spaces at different positions.
- **Positional Encoding**: Added to give the model position information since there
  is no recurrence or convolution.

## Architecture Details

The Transformer uses an encoder-decoder structure. The encoder maps input symbols
to continuous representations using 6 identical layers, each with multi-head
self-attention and a feed-forward network. The decoder also has 6 layers, adding
a third sub-layer for cross-attention over the encoder output. Residual connections
and layer normalization are applied around each sub-layer.

## Attention Function

The attention function maps queries Q, keys K, and values V:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Multi-head attention runs h parallel attention heads and concatenates the results.

## Results

The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German translation,
improving over existing best results by over 2 BLEU. On English-to-French, it
achieves 41.0 BLEU, outperforming all previously published single models.
"""

WIKI_LORA = """\
---
title: "LoRA: Low-Rank Adaptation"
authors: Hu et al.
year: 2021
tags: lora, fine-tuning, parameter-efficient
---

# LoRA: Low-Rank Adaptation

LoRA freezes pre-trained model weights and injects trainable low-rank decomposition
matrices into each [[Transformer]] layer, greatly reducing trainable parameters.

## Method

For a weight matrix W0, LoRA constrains the update: W0 + delta_W = W0 + BA,
where B and A are low-rank matrices with rank r << min(d, k). During training,
W0 is frozen. A is initialized with random Gaussian, B with zeros.

## Key Benefits

- **10,000x fewer parameters**: For GPT-3 175B, LoRA reduces trainable parameters
  from 175 billion to ~18 million.
- **3x less GPU memory**: No need to store optimizer states for the full model.
- **No inference latency**: The low-rank matrices can be merged with frozen weights.
- **Switchable**: Multiple LoRA adapters can be swapped at inference time.

## Hyperparameters

- Rank (r): Typically 4-64. r=8 is a good default.
- Alpha: Scaling factor applied as alpha/r.
- Target modules: Usually attention projections (q_proj, v_proj).
"""

WIKI_SELF_ATTENTION_ENTITY = """\
---
title: Self-Attention
type: entity
---

# Self-Attention

Self-attention (also called intra-attention) relates different positions of a
single sequence to compute a representation. Each position attends to all positions
in the previous layer. The computation uses three projections: queries (Q),
keys (K), and values (V), all derived from the input.

The output is a weighted sum of values, where weights are determined by the
compatibility between queries and keys via scaled dot-product.
"""

WIKI_TRANSFORMER_ENTITY = """\
---
title: Transformer
type: entity
---

# Transformer

The Transformer is a neural network architecture introduced in "Attention Is All
You Need" (Vaswani et al., 2017). It relies entirely on self-attention mechanisms
instead of recurrence or convolution, enabling much greater parallelization during
training.

Key components: multi-head self-attention, position-wise feed-forward networks,
residual connections, layer normalization, and positional encodings.
"""

WIKI_GPT3_LATE = """\
---
title: GPT-3 Few-Shot Learning
authors: Brown et al.
year: 2020
tags: gpt-3, few-shot, scaling
---

# GPT-3: Language Models are Few-Shot Learners

GPT-3 demonstrates that scaling up language models greatly improves few-shot
performance. The 175B parameter model can perform tasks by conditioning on
examples in the prompt, without any gradient updates.

## In-Context Learning

- Zero-shot: Only a task description.
- One-shot: One example alongside the description.
- Few-shot: 10-100 examples in the prompt.

Performance improves log-linearly with model scale across all settings.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════


def _header(text: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def _subheader(text: str) -> None:
    print(f"\n--- {text} ---\n")


def _ask_questions(model: KnowledgeEnhancedModel, questions: list[str]) -> list[str]:
    answers = []
    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q}")
        ans = model.answer(q)
        for line in ans.splitlines():
            print(f"      {line}")
        print()
        answers.append(ans)
    return answers


# ═══════════════════════════════════════════════════════════════════════════
# Main scenario
# ═══════════════════════════════════════════════════════════════════════════


def run_scenario(wiki_dir: Path, *, verbose: bool = False) -> dict[str, Any]:
    """Run the full end-to-end scenario. Returns results dict for testing."""
    results: dict[str, Any] = {"steps": []}

    sources_dir = wiki_dir / "sources"
    entities_dir = wiki_dir / "entities"
    sources_dir.mkdir(parents=True, exist_ok=True)
    entities_dir.mkdir(parents=True, exist_ok=True)

    generator = ContentAwareGenerator()
    repository = InMemoryAdapterRepository()
    model = KnowledgeEnhancedModel(repository)

    watcher_config = WatcherConfig(directory=wiki_dir)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 -- Baseline: empty wiki, no adapters
    # ══════════════════════════════════════════════════════════════════
    _header("STEP 1: Baseline -- Empty Wiki (No Adapters)")
    print(f"  Wiki directory: {wiki_dir}")
    print(f"  Adapters loaded: {repository.adapter_count}")
    print()

    step1_answers = _ask_questions(model, QUESTIONS)
    results["steps"].append(
        {
            "name": "baseline",
            "adapters": repository.adapter_count,
            "answers": step1_answers,
        }
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 -- Add Transformer paper + entity pages to wiki
    # ══════════════════════════════════════════════════════════════════
    _header("STEP 2: Adding 'Attention Is All You Need' paper to wiki")

    (sources_dir / "attention-paper.md").write_text(WIKI_TRANSFORMER_V1, encoding="utf-8")
    (entities_dir / "self-attention.md").write_text(WIKI_SELF_ATTENTION_ENTITY, encoding="utf-8")
    (entities_dir / "transformer.md").write_text(WIKI_TRANSFORMER_ENTITY, encoding="utf-8")

    print("  Written: sources/attention-paper.md")
    print("  Written: entities/self-attention.md")
    print("  Written: entities/transformer.md")

    source = WikiKnowledgeSource(watcher_config, aggregate=True)
    orchestrator = KnowledgeFusionOrchestrator(
        source=source, generator=generator, repository=repository
    )
    manifests = orchestrator.compile_all()

    print(f"\n  Pipeline triggered! Generated {len(manifests)} adapter(s):")
    for m in manifests:
        h = repository.get_content_hash(m.adapter_id)
        print(f"    -> {m.adapter_id}  [hash={h}]")
    print()

    _subheader("Asking the same questions again")
    step2_answers = _ask_questions(model, QUESTIONS)
    results["steps"].append(
        {
            "name": "transformer_paper",
            "adapters": repository.adapter_count,
            "manifests": [m.adapter_id for m in manifests],
            "answers": step2_answers,
        }
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 -- Add LoRA paper to wiki
    # ══════════════════════════════════════════════════════════════════
    _header("STEP 3: Adding 'LoRA' paper to wiki")

    (sources_dir / "lora-paper.md").write_text(WIKI_LORA, encoding="utf-8")
    print("  Written: sources/lora-paper.md")

    source_s3 = WikiKnowledgeSource(watcher_config, aggregate=True)
    orch_s3 = KnowledgeFusionOrchestrator(
        source=source_s3, generator=generator, repository=repository
    )
    manifests = orch_s3.compile_all()

    print(f"\n  Pipeline triggered! Generated {len(manifests)} adapter(s):")
    for m in manifests:
        h = repository.get_content_hash(m.adapter_id)
        print(f"    -> {m.adapter_id}  [hash={h}]")
    print()

    _subheader("Asking the same questions again")
    step3_answers = _ask_questions(model, QUESTIONS)
    results["steps"].append(
        {
            "name": "lora_paper_added",
            "adapters": repository.adapter_count,
            "manifests": [m.adapter_id for m in manifests],
            "answers": step3_answers,
        }
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 -- Update the Transformer page with more detail
    # ══════════════════════════════════════════════════════════════════
    _header("STEP 4: Updating Transformer page with architecture details")

    old_hash = repository.get_content_hash("sources/attention-paper")
    (sources_dir / "attention-paper.md").write_text(WIKI_TRANSFORMER_V2_UPDATE, encoding="utf-8")
    print("  Updated: sources/attention-paper.md")
    print(f"  Old content hash: {old_hash}")

    source_s4 = WikiKnowledgeSource(watcher_config, aggregate=True)
    orch_s4 = KnowledgeFusionOrchestrator(
        source=source_s4, generator=generator, repository=repository
    )
    manifests = orch_s4.compile_all()

    new_hash = repository.get_content_hash("sources/attention-paper")
    print(f"  New content hash: {new_hash}")
    hash_changed = old_hash != new_hash
    print(f"  Content changed:  {hash_changed}")
    print(f"\n  Pipeline triggered! Regenerated {len(manifests)} adapter(s):")
    for m in manifests:
        h = repository.get_content_hash(m.adapter_id)
        print(f"    -> {m.adapter_id}  [hash={h}]")
    print()

    _subheader("Asking the same questions again (adapter updated)")
    step4_answers = _ask_questions(model, QUESTIONS)
    results["steps"].append(
        {
            "name": "transformer_updated",
            "adapters": repository.adapter_count,
            "hash_changed": hash_changed,
            "answers": step4_answers,
        }
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 5 -- Live watcher: drop a new file, auto-detected
    # ══════════════════════════════════════════════════════════════════
    _header("STEP 5: Live Watcher -- auto-detect new wiki page")

    source_live = WikiKnowledgeSource(watcher_config, aggregate=True)
    orch_live = KnowledgeFusionOrchestrator(
        source=source_live, generator=generator, repository=repository
    )

    detected_docs: list[str] = []
    original_process = orch_live.process_document

    def tracking_process(ctx: DocumentContext) -> AdapterManifest:
        detected_docs.append(ctx.document_id)
        return original_process(ctx)

    orch_live.process_document = tracking_process  # type: ignore[assignment]

    with orch_live:
        print("  Watcher started. Dropping new file...")
        time.sleep(0.3)

        (sources_dir / "gpt3-paper.md").write_text(WIKI_GPT3_LATE, encoding="utf-8")
        print("  Written: sources/gpt3-paper.md")

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if "sources/gpt3-paper" in detected_docs:
                break
            time.sleep(0.2)

    watcher_detected = "sources/gpt3-paper" in detected_docs
    if watcher_detected:
        print("\n  Watcher detected new file and processed it automatically!")
        h = repository.get_content_hash("sources/gpt3-paper")
        print(f"    -> sources/gpt3-paper  [hash={h}]")
    else:
        print("\n  (Watcher detection timed out -- running batch fallback)")
        source_fb = WikiKnowledgeSource(watcher_config, aggregate=True)
        orch_fb = KnowledgeFusionOrchestrator(
            source=source_fb, generator=generator, repository=repository
        )
        orch_fb.compile_all()

    print(f"  Total adapters: {repository.adapter_count}")
    print()

    _subheader("Final answers with all knowledge loaded")
    step5_answers = _ask_questions(model, QUESTIONS)
    results["steps"].append(
        {
            "name": "live_watcher",
            "adapters": repository.adapter_count,
            "watcher_detected": watcher_detected,
            "answers": step5_answers,
        }
    )

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    _header("SUMMARY -- Knowledge Progression")

    print(
        f"\n"
        f"  Step 1 (Baseline):     0 adapters"
        f" -> generic 'I don't know' answers\n"
        f"  Step 2 (+Transformer): {results['steps'][1]['adapters']} adapters"
        f" -> answers about attention, transformers\n"
        f"  Step 3 (+LoRA):        {results['steps'][2]['adapters']} adapters"
        f" -> also answers about LoRA, fine-tuning\n"
        f"  Step 4 (Update):       {results['steps'][3]['adapters']} adapters"
        f" -> richer transformer architecture details\n"
        f"  Step 5 (Live):         {results['steps'][4]['adapters']} adapters"
        f" -> auto-detected new GPT-3 paper\n"
        f"\n"
        f"  The model's answers progressively improved as wiki content was added.\n"
        f"  Each wiki change triggered the pipeline:"
        f" WikiSource -> Generator -> Repository.\n"
        f"  The adapter content was immediately available for enhanced inference.\n"
    )

    results["total_adapters"] = repository.adapter_count
    results["all_adapter_ids"] = list(repository.get_all_content().keys())
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E2E scenario demo")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--wiki-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "demo_wiki",
    )
    parser.add_argument("--clean", action="store_true", help="Remove wiki dir first.")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s | %(name)s | %(message)s")

    import shutil

    wiki_dir = args.wiki_dir.resolve()
    if args.clean and wiki_dir.exists():
        shutil.rmtree(wiki_dir)

    run_scenario(wiki_dir)


if __name__ == "__main__":
    main()
