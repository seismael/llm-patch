"""Gemini E2E comparison — before/after wiki knowledge.

Strategy: manually populate wiki pages from paper content to avoid
the 15+ API calls needed for agent-based ingestion. Only 2 Gemini
API calls are needed: one BEFORE (raw) and one AFTER (wiki-enhanced).
"""

import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# ── Load .env ──────────────────────────────────────────────────
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import logging  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

import litellm  # noqa: E402

from llm_patch.wiki.agents.litellm_agent import LiteLLMWikiAgent  # noqa: E402
from llm_patch.wiki.index import IndexEntry, WikiIndex  # noqa: E402
from llm_patch.wiki.manager import WikiManager  # noqa: E402
from llm_patch.wiki.page import (  # noqa: E402
    ConfidenceLevel,
    PageType,
    WikiPage,
    WikiPageFrontmatter,
)

litellm.suppress_debug_info = True

RAW_PAPERS = Path(__file__).resolve().parents[1] / "examples" / "raw" / "papers"
OUTPUT_FILE = Path(__file__).resolve().parent / "comparison_results.txt"
QUESTION = (
    "What is the Transformer architecture and how does self-attention work? "
    "How does LoRA enable efficient fine-tuning?"
)


class DualWriter:
    """Write to both stdout and a file."""

    def __init__(self, filepath: Path):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> int:
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


# ── Pre-built wiki pages (avoids ~15 LLM calls for ingestion) ───
WIKI_PAGES = [
    {
        "path": "entities/transformer.md",
        "title": "Transformer",
        "type": PageType.ENTITY,
        "tags": ["transformer", "architecture", "attention", "encoder-decoder"],
        "body": (
            "# Transformer\n\n"
            "The **Transformer** is a neural network architecture introduced by "
            "Vaswani et al. (2017) in *Attention Is All You Need*. It dispenses "
            "with recurrence and convolutions entirely, relying solely on "
            "**self-attention** mechanisms.\n\n"
            "## Architecture\n\n"
            "The Transformer follows an **encoder-decoder** structure:\n\n"
            "- **Encoder**: Each layer has two sub-layers — a multi-head "
            "self-attention mechanism and a position-wise feed-forward network. "
            "Residual connections and layer normalization surround each sub-layer.\n"
            "- **Decoder**: Adds a third sub-layer performing multi-head attention "
            "over the encoder output.\n\n"
            "## Impact\n\n"
            "The Transformer became the foundation for virtually all modern LLMs "
            "including BERT, GPT, T5, and their successors. See also: "
            "[[Self-Attention]], [[Multi-Head Attention]], [[Positional Encoding]]."
        ),
    },
    {
        "path": "concepts/self-attention.md",
        "title": "Self-Attention",
        "type": PageType.CONCEPT,
        "tags": ["self-attention", "attention", "transformer", "scaled-dot-product"],
        "body": (
            "# Self-Attention\n\n"
            "**Self-attention** (also called intra-attention) computes attention "
            "scores between every pair of positions in a sequence, allowing the "
            "model to capture long-range dependencies without recurrence.\n\n"
            "## Scaled Dot-Product Attention\n\n"
            "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}"
            "{\\sqrt{d_k}}\\right)V$$\n\n"
            "Queries (Q), keys (K), and values (V) are linear projections of the "
            "input. The scaling factor $\\sqrt{d_k}$ prevents dot products from "
            "growing too large.\n\n"
            "## Multi-Head Attention\n\n"
            "$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, "
            "\\ldots, \\text{head}_h)W^O$$\n\n"
            "Multiple attention heads allow the model to attend to information "
            "from different representation sub-spaces at different positions. "
            "See also: [[Transformer]], [[Positional Encoding]]."
        ),
    },
    {
        "path": "concepts/positional-encoding.md",
        "title": "Positional Encoding",
        "type": PageType.CONCEPT,
        "tags": ["positional-encoding", "transformer", "sinusoidal"],
        "body": (
            "# Positional Encoding\n\n"
            "Since the [[Transformer]] contains no recurrence and no convolution, "
            "**positional encodings** are added to give the model information about "
            "the relative or absolute position of tokens in the sequence.\n\n"
            "The original paper uses sinusoidal functions of different frequencies "
            "to encode positions, allowing the model to extrapolate to sequence "
            "lengths not seen during training."
        ),
    },
    {
        "path": "entities/lora.md",
        "title": "LoRA",
        "type": PageType.ENTITY,
        "tags": ["lora", "fine-tuning", "parameter-efficient", "low-rank", "adaptation"],
        "body": (
            "# LoRA: Low-Rank Adaptation\n\n"
            "**LoRA** (Hu et al., 2021) is a parameter-efficient fine-tuning "
            "technique that freezes pre-trained model weights and injects "
            "trainable **low-rank decomposition matrices** into each layer.\n\n"
            "## Method\n\n"
            "For a pre-trained weight matrix $W_0$, LoRA constrains the update:\n\n"
            "$$W_0 + \\Delta W = W_0 + BA$$\n\n"
            "where $B \\in \\mathbb{R}^{d \\times r}$ and "
            "$A \\in \\mathbb{R}^{r \\times k}$ with rank $r \\ll \\min(d, k)$.\n\n"
            "- $W_0$ is **frozen** (no gradient updates)\n"
            "- $A$ initialized with random Gaussian, $B$ initialized to zero\n"
            "- Modified forward pass: $h = W_0 x + BAx$\n\n"
            "## Key Hyperparameters\n\n"
            "- **Rank ($r$)**: Typically 4–64; $r=8$ is a common default\n"
            "- **Alpha ($\\alpha$)**: Scaling factor applied as $\\alpha/r$\n"
            "- **Target Modules**: Usually attention projections ($W_q$, $W_v$)\n\n"
            "## Benefits\n\n"
            "- **10,000x fewer trainable parameters** than full fine-tuning (GPT-3)\n"
            "- **3x less GPU memory** required\n"
            "- **Zero additional inference latency** — trained matrices merge with "
            "frozen weights at deployment\n"
            "- Matches or exceeds full fine-tuning quality on RoBERTa, DeBERTa, "
            "GPT-2, and GPT-3\n\n"
            "## Impact\n\n"
            "LoRA became the de facto standard for parameter-efficient fine-tuning. "
            "The PEFT library (Hugging Face) implements LoRA, QLoRA, and other "
            "variants. See also: [[Transformer]], [[Self-Attention]]."
        ),
    },
    {
        "path": "summaries/attention-is-all-you-need.md",
        "title": "Attention Is All You Need",
        "type": PageType.SUMMARY,
        "tags": ["transformer", "attention", "vaswani", "2017"],
        "body": (
            "# Summary: Attention Is All You Need (Vaswani et al., 2017)\n\n"
            "Proposes the [[Transformer]] architecture that relies solely on "
            "[[Self-Attention]] mechanisms, dispensing with recurrence and "
            "convolutions. Achieves 28.4 BLEU on WMT 2014 EN-DE, improving "
            "over existing best by 2+ BLEU. The architecture uses "
            "[[Multi-Head Attention]], [[Positional Encoding]], and an "
            "encoder-decoder structure with residual connections and layer "
            "normalization."
        ),
    },
    {
        "path": "summaries/lora-low-rank-adaptation.md",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "type": PageType.SUMMARY,
        "tags": ["lora", "fine-tuning", "hu", "2021"],
        "body": (
            "# Summary: LoRA (Hu et al., 2021)\n\n"
            "Introduces [[LoRA]], a parameter-efficient fine-tuning method that "
            "freezes pre-trained weights and injects trainable low-rank matrices "
            "($\\Delta W = BA$). Reduces trainable parameters by 10,000x for "
            "GPT-3 175B while matching full fine-tuning quality. Zero inference "
            "latency overhead since trained matrices merge with frozen weights."
        ),
    },
]


def build_wiki(wiki_dir: Path, index: WikiIndex) -> int:
    """Create wiki pages and index entries from pre-built content."""
    count = 0
    for spec in WIKI_PAGES:
        page = WikiPage(
            frontmatter=WikiPageFrontmatter(
                title=spec["title"],
                type=spec["type"],
                tags=spec["tags"],
                sources=["examples/raw/papers/"],
                confidence=ConfidenceLevel.HIGH,
            ),
            body=spec["body"],
            path=spec["path"],
        )
        page_path = wiki_dir / spec["path"]
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(page.to_markdown(), encoding="utf-8")

        category = spec["type"].value + "s"
        index.add_entry(
            IndexEntry(
                path=spec["path"],
                title=spec["title"],
                summary=spec["body"][:100].replace("\n", " ").strip(),
                tags=spec["tags"],
                category=category,
            )
        )
        count += 1
    index.save()
    return count


def main() -> None:
    print(f"[init] Gemini API key present: {bool(os.environ.get('GEMINI_API_KEY'))}")
    agent = LiteLLMWikiAgent(model="gemini/gemini-2.0-flash", max_tokens=8192)
    print("[init] Agent created (gemini-2.0-flash)")

    # ─── BEFORE: raw LLM answer (no wiki context) ───────────────
    print()
    print("=" * 60)
    print("BEFORE: Raw LLM answer (no wiki knowledge)")
    print("=" * 60)
    raw_answer = agent._call(
        "You are a helpful assistant. Answer concisely in 3-5 sentences.",
        QUESTION,
    )
    print(raw_answer)
    print()
    print(f"[info] Before answer: {len(raw_answer)} chars")

    # ─── BUILD WIKI (manual — 0 API calls) ──────────────────────
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # Copy raw sources
        raw_dest = td_path / "raw" / "papers"
        raw_dest.mkdir(parents=True)
        for name in ("attention-is-all-you-need.md", "lora-low-rank-adaptation.md"):
            shutil.copy2(RAW_PAPERS / name, raw_dest / name)
            print(f"[build] Copied: {name}")

        # Initialize wiki structure
        mgr = WikiManager(agent=agent, base_dir=td_path)
        mgr.init()
        print("[build] Wiki initialized")

        # Populate wiki pages directly (no LLM calls)
        wiki_dir = td_path / "wiki"
        n = build_wiki(wiki_dir, mgr.index)
        print(f"[build] Created {n} wiki pages (manual, 0 API calls)")

        status = mgr.status()
        print(f"[build] Status: {status}")

        # ─── AFTER: wiki-enhanced answer (1 API call) ──────────
        print()
        print("=" * 60)
        print("AFTER: Wiki-enhanced answer (with compiled knowledge)")
        print("=" * 60)
        result = mgr.query(QUESTION)
        print(result.answer)
        print(f"\n[result] Cited pages: {result.cited_pages}")

        # ─── COMPARISON ────────────────────────────────────────
        print()
        print("=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Question:             {QUESTION[:80]}...")
        print(f"Before answer length: {len(raw_answer)} chars")
        print(f"After answer length:  {len(result.answer)} chars")
        print(f"Citations:            {len(result.cited_pages)} pages")
        print(f"Wiki pages compiled:  {status.get('wiki_pages', 0)}")
        print(f"Index entries:        {status.get('index_entries', 0)}")

        wiki_terms = ["transformer", "self-attention", "lora", "multi-head", "positional"]
        before_hits = sum(1 for t in wiki_terms if t.lower() in raw_answer.lower())
        after_hits = sum(1 for t in wiki_terms if t.lower() in result.answer.lower())
        print(f"Domain terms (before): {before_hits}/{len(wiki_terms)}")
        print(f"Domain terms (after):  {after_hits}/{len(wiki_terms)}")
        if after_hits > before_hits:
            print(">>> Wiki-enhanced answer has MORE domain specificity!")
        if result.cited_pages:
            print(">>> Wiki-enhanced answer includes CITATIONS to source pages!")


if __name__ == "__main__":
    writer = DualWriter(OUTPUT_FILE)
    sys.stdout = writer
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(f"\n\nERROR:\n{tb}")
        sys.stderr.write(tb)
        sys.exit(1)
    finally:
        sys.stdout = writer.stdout
        writer.close()
        print(f"\nResults saved to: {OUTPUT_FILE}")
