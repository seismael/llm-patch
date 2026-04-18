# Wiki Schema (CLAUDE.md-style)

This file defines the structure, conventions, and rules for the LLM Wiki.
Load it with `WikiSchema.from_file("wiki_schema.md")`.

## Directory Layout

- `raw/` — Immutable raw source documents (papers, notes, articles)
- `wiki/` — LLM-maintained knowledge base
  - `summaries/` — One page per raw source
  - `concepts/` — Distilled concept pages
  - `entities/` — Named entities (people, tools, architectures)
  - `syntheses/` — Cross-cutting analysis pages
  - `journal/` — Dated observation logs
  - `index.md` — Master catalog of all pages
  - `log.md` — Chronological activity log

## Page Format

Every wiki page must have YAML frontmatter:

```yaml
---
title: Page Title
type: concept|entity|summary|synthesis|journal
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [raw/papers/example.md]
confidence: high|medium|low|uncertain
---
```

## Page Types

### summary
One-page distillation of a raw source document.
Required sections: Key Points, Relevant Concepts, Source Metadata

### concept
Distilled concept page — definition, context, and relationships.
Required sections: Definition, Context, Related Concepts

### entity
Named entity page — person, tool, architecture, dataset.
Required sections: Overview, Key Contributions

### synthesis
Cross-cutting analysis combining multiple sources.
Required sections: Thesis, Evidence, Open Questions

### journal
Dated observation or research note.
Required sections: Observation, Context

## Rules

1. Never modify files in raw/ — they are immutable source-of-truth
2. Always update index.md after creating/removing a page
3. Use [[wikilinks]] for cross-references between pages
4. Every page must have YAML frontmatter with at least title and type
5. Summaries must link back to their source document
6. Flag contradictions between sources explicitly
7. Use confidence levels: high (verified), medium (inferred), low (speculative)
8. Log every operation in log.md
