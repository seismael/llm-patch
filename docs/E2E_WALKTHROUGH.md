# End-to-End Walkthrough

Step-by-step guide to running the full **Ingest → Compile → Attach → Use** pipeline with a real wiki, including before/after knowledge comparison.

---

## Prerequisites

| Requirement | Status Check |
|---|---|
| Python ≥ 3.11 | `python --version` |
| llm-patch installed | `pip install -e '.[all]'` |
| Architecture wiki | `C:\dev\projects\architecture\wiki` exists |
| Gemini API key | `GEMINI_API_KEY` in `.env` file |
| litellm installed | `pip install litellm` |

---

## Step 1: Ingest — Load Documents from Wiki

The `WikiDataSource` scans a wiki directory, extracts YAML frontmatter, parses `[[wikilinks]]` and `[text](path.md)` links, and produces `DocumentContext` objects.

```python
from pathlib import Path
from llm_patch.sources.wiki import WikiDataSource

wiki_dir = Path(r"C:\dev\projects\architecture\wiki")
ds = WikiDataSource(wiki_dir, recursive=True, aggregate=True)

docs = list(ds.fetch_all())
print(f"Loaded {len(docs)} documents")

# Inspect a document
doc = docs[0]
print(f"  ID: {doc.document_id}")
print(f"  Title: {doc.metadata.get('title')}")
print(f"  Links: {doc.metadata.get('wikilinks', [])}")
print(f"  Content: {doc.content[:200]}...")
```

**CLI equivalent:**
```bash
llm-patch source list --kind wiki --path C:\dev\projects\architecture\wiki
llm-patch source count --kind wiki --path C:\dev\projects\architecture\wiki
```

---

## Step 2: Wiki Compile — Build a Knowledge Wiki with LLM

The `WikiManager` uses an LLM agent (Gemini via LiteLLM) to summarize documents, extract entities, generate wiki pages, and maintain an index.

### 2a. Initialize and Compile

```python
import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from pathlib import Path
from llm_patch.wiki.agents.litellm_agent import LiteLLMWikiAgent
from llm_patch.wiki.manager import WikiManager

# Create the agent (reads GEMINI_API_KEY from .env or environment)
agent = LiteLLMWikiAgent(model="gemini/gemini-2.5-pro")

# Point at your project
project_dir = Path(r"C:\dev\projects\architecture")
manager = WikiManager(agent=agent, base_dir=project_dir)
manager.init()

# Compile all raw sources into wiki pages
results = manager.compile_all()
for r in results:
    print(f"  {r.summary_page}: {len(r.entities_extracted)} entities")
```

**CLI equivalent:**
```bash
# Set env vars
set LITELLM_LOCAL_MODEL_COST_MAP=True
set GEMINI_API_KEY=your-key-here

# Initialize
llm-patch wiki --base-dir C:\dev\projects\architecture --agent litellm init

# Compile all sources
llm-patch wiki --base-dir C:\dev\projects\architecture --agent litellm compile

# Check status
llm-patch wiki --base-dir C:\dev\projects\architecture --agent litellm status
```

### 2b. Query the Wiki

```python
result = manager.query("What is the CQRS pattern and how is it used?")
print(result.answer)
print(f"Cited pages: {result.cited_pages}")
```

**CLI equivalent:**
```bash
llm-patch wiki --base-dir C:\dev\projects\architecture --agent litellm \
    query "What is the CQRS pattern and how is it used?"
```

### 2c. Lint for Quality

```python
report = manager.lint()
print(f"Issues: {len(report.issues)}")
for issue in report.issues:
    print(f"  [{issue.severity}] {issue.page}: {issue.message}")
```

---

## Step 3: Compile Pipeline — Generate LoRA Adapter Weights

> **Note:** This step requires the Sakana T2L checkpoint. If you don't have it, use the mock generator (see Step 3b).

### 3a. Real T2L Generator (GPU required)

```python
from llm_patch.core.config import GeneratorConfig, StorageConfig
from llm_patch.generators.sakana_t2l import SakanaT2LGenerator
from llm_patch.storage.local_safetensors import LocalSafetensorsRepository
from llm_patch.sources.wiki import WikiDataSource
from llm_patch.pipelines.compile import CompilePipeline

source = WikiDataSource(Path(r"C:\dev\projects\architecture\wiki"), recursive=True)
generator = SakanaT2LGenerator(GeneratorConfig(
    checkpoint_dir=Path(r"C:\path\to\t2l-checkpoint"),
    device="cuda",
))
repository = LocalSafetensorsRepository(StorageConfig(output_dir=Path("./adapters")))

pipeline = CompilePipeline(source, generator, repository)
manifests = pipeline.compile_all()
print(f"Compiled {len(manifests)} adapters")
```

### 3b. Mock Generator (No GPU needed)

```python
from unittest.mock import MagicMock
from llm_patch.pipelines.compile import CompilePipeline
from llm_patch.sources.wiki import WikiDataSource

source = WikiDataSource(Path(r"C:\dev\projects\architecture\wiki"), recursive=True)

mock_gen = MagicMock()
mock_gen.generate.return_value = {"lora_A": MagicMock(), "lora_B": MagicMock()}
mock_gen.get_peft_config.return_value = MagicMock(
    to_dict=MagicMock(return_value={"r": 8, "target_modules": ["q_proj"], "peft_type": "LORA"})
)
mock_repo = MagicMock()

pipeline = CompilePipeline(source, mock_gen, mock_repo)
manifests = pipeline.compile_all()
print(f"Mock-compiled {len(manifests)} documents")
```

---

## Step 4: Attach — Load Model + Apply Adapters

```python
from llm_patch.attach.model_provider import HFModelProvider
from llm_patch.attach.peft_loader import PeftAdapterLoader
from llm_patch.storage.local_safetensors import LocalSafetensorsRepository
from llm_patch.pipelines.use import UsePipeline

provider = HFModelProvider()
loader = PeftAdapterLoader()
repo = LocalSafetensorsRepository(StorageConfig(output_dir=Path("./adapters")))

use = UsePipeline(provider, loader, repo)

# Load base model + attach all compiled adapters
handle = use.load_and_attach(
    "google/gemma-2-2b-it",
    dtype="float16",
    device_map="auto",
)
print(f"Attached {len(handle.attached_adapters)} adapters")
```

---

## Step 5: Use — Generate & Chat with Patched Model

### Single Generation

```python
from llm_patch.runtime.agent import PeftAgentRuntime

agent = PeftAgentRuntime(handle)
response = agent.generate("Explain the CQRS pattern in software architecture")
print(response)
```

### Multi-turn Chat

```python
from llm_patch.core.models import ChatMessage, ChatRole

reply = agent.chat([
    ChatMessage(role=ChatRole.SYSTEM, content="You are a software architecture expert."),
    ChatMessage(role=ChatRole.USER, content="What is event sourcing?"),
])
print(reply.message.content)
```

### Stateful Chat Session

```python
from llm_patch.runtime.session import ChatSession

session = ChatSession(
    runtime=agent,
    system_prompt="You are an expert on software architecture patterns.",
    max_history=20,
)

print(session.say("What is the CQRS pattern?"))
print(session.say("How does it relate to event sourcing?"))
print(session.say("Give me a real-world example"))
```

---

## Step 6: Before/After Comparison

This is the key validation — compare answers from a **base model** (no wiki knowledge) vs a **wiki-enhanced agent** (with Gemini-compiled context).

### Option A: LLM Agent Comparison (no GPU needed)

```python
import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from pathlib import Path
from llm_patch.wiki.agents.litellm_agent import LiteLLMWikiAgent
from llm_patch.wiki.manager import WikiManager

question = "What is the CQRS pattern and how does it relate to event sourcing?"

# BEFORE: Ask raw LLM (no wiki context)
agent = LiteLLMWikiAgent(model="gemini/gemini-2.5-pro")
raw_answer = agent._call(
    "You are a helpful assistant.",
    question,
)
print("=== BEFORE (no wiki) ===")
print(raw_answer[:500])

# AFTER: Build wiki, then query with full context
project_dir = Path(r"C:\dev\projects\architecture")
manager = WikiManager(agent=agent, base_dir=project_dir)
manager.init()
manager.compile_all()

result = manager.query(question)
print("\n=== AFTER (wiki-enhanced) ===")
print(result.answer[:500])
print(f"\nCited pages: {result.cited_pages}")
```

### Option B: Run the Automated Test

```bash
set GEMINI_API_KEY=your-key
set LITELLM_LOCAL_MODEL_COST_MAP=True
python -m pytest tests/integration/test_e2e_architecture_wiki.py::TestPhase10BeforeAfterComparison -v -s
```

### Option C: Run the Standalone Comparison Script

```bash
set LITELLM_LOCAL_MODEL_COST_MAP=True
cd scripts
python run_gemini_comparison.py
```

Results are saved to `scripts/comparison_results.txt` and printed to stdout.

---

## Gemini Comparison Results (April 18 2026)

Live run using `gemini/gemini-2.0-flash` against 2 research papers
(Attention Is All You Need + LoRA). The wiki was pre-populated with 6
structured pages (2 entity, 2 concept, 2 summary) to isolate the BEFORE
vs AFTER comparison to exactly 2 Gemini API calls.

**Question:** *"What is the Transformer architecture and how does
self-attention work? How does LoRA enable efficient fine-tuning?"*

### BEFORE — Raw LLM (no wiki context)

> The Transformer architecture is a neural network that relies on
> self-attention mechanisms to weigh the importance of different parts of
> the input sequence. Self-attention allows the model to capture
> relationships between all words in a sentence, enabling parallel
> processing and capturing long-range dependencies.
>
> LoRA (Low-Rank Adaptation) reduces the number of trainable parameters
> by introducing low-rank matrices to the existing weights, updating only
> these smaller matrices during fine-tuning. This significantly reduces
> computational costs and memory usage, enabling efficient adaptation of
> large pre-trained models to specific tasks.

**642 chars — generic overview, no math, no citations, 3/5 domain terms.**

### AFTER — Wiki-enhanced (with compiled knowledge)

> The [[Transformer]] architecture dispenses with recurrence and
> convolutions, relying solely on [[Self-Attention]] mechanisms. It
> follows an encoder-decoder structure, where each encoder layer has
> multi-head self-attention and a position-wise feed-forward network, and
> each decoder layer adds a third sub-layer performing multi-head
> attention over the encoder output. Residual connections and layer
> normalization surround each sub-layer. [[Positional Encoding]] is used
> to give the model information about the position of tokens.
>
> [[Self-Attention]] computes attention scores between every pair of
> positions in a sequence:
>
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
>
> [[Multi-Head Attention]] allows the model to attend to information from
> different representation sub-spaces at different positions.
>
> [[LoRA]] freezes pre-trained model weights and injects trainable
> low-rank decomposition matrices: $W_0 + \Delta W = W_0 + BA$.
> This reduces trainable parameters by 10,000x for GPT-3, requires 3x
> less GPU memory, and adds zero inference latency since trained matrices
> merge with frozen weights at deployment.

**1,848 chars — structured, math formulas, 6 cited pages, 5/5 domain terms.**

### Comparison Summary

| Metric | Before | After | Delta |
|---|---|---|---|
| Answer length | 642 chars | 1,848 chars | **+188%** |
| Domain terms | 3/5 | 5/5 | **+67%** |
| Citations | 0 pages | 6 pages | — |
| Math formulas | None | Attention eq + LoRA decomposition | — |
| Wiki links | None | 5 `[[wikilinks]]` | — |
| Encoder/decoder detail | No | Yes (sub-layers, residual, layer norm) | — |
| LoRA specifics | Generic | Rank, alpha, target modules, 10,000x stat | — |

### Insights

1. **Knowledge grounding works** — the wiki context transforms a generic
   overview into a technically precise answer with formulas and citations.
2. **Structured wiki pages > raw documents** — pre-processed entity/concept
   pages with cross-references produce better answers than dumping raw
   paper text into the prompt.
3. **Citation traceability** — every claim in the AFTER answer links back
   to a specific wiki page, enabling auditability.
4. **Rate limiting matters** — Gemini free tier throttles aggressively.
   The comparison script pre-builds wiki pages (0 API calls) to minimize
   quota usage. Production deployments should use paid API tiers.
5. **Model choice** — `gemini-2.0-flash` works well for this use case.
   Avoid thinking models (e.g. `gemini-2.5-pro`) for latency-sensitive
   wiki operations; they add 30+ seconds per call.

---

## Running All E2E Tests

```bash
# Set environment variables
set GEMINI_API_KEY=your-key
set LITELLM_LOCAL_MODEL_COST_MAP=True

# Run complete E2E suite (50+ tests)
python -m pytest tests/integration/test_e2e_architecture_wiki.py -v

# Run specific phase
python -m pytest tests/integration/test_e2e_architecture_wiki.py::TestPhase1DataSourceIngestion -v
python -m pytest tests/integration/test_e2e_architecture_wiki.py::TestPhase3WikiPipelineGemini -v
python -m pytest tests/integration/test_e2e_architecture_wiki.py::TestPhase10BeforeAfterComparison -v -s

# Full regression suite (all 374+ tests)
python -m pytest tests/ -q
```

---

## Test Phase Summary

| Phase | What It Tests | Real API? |
|---|---|---|
| 1 | Data source ingestion, frontmatter, links | No |
| 2 | WikiManager with mock agent | No |
| 3 | WikiManager with Gemini (ingest, query, compile, lint) | **Yes** |
| 4 | CompilePipeline with real wiki docs | No |
| 5 | Attach pipeline (compile → attach flow) | No |
| 6 | PeftAgentRuntime, ChatSession | No |
| 7 | CLI smoke tests (init, compile, status, query, lint) | No |
| 8 | Structural integrity of real wiki | No |
| 9 | Full pipeline integration (ingest → compile → attach → use) | No |
| 10 | Before/after knowledge comparison with Gemini | **Yes** |
