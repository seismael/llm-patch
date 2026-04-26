# Quickstart example — minimal sources for `llm-patch compile`

This folder is the backing data for the README cast and
[docs/QUICKSTART.md](../../docs/QUICKSTART.md). It is intentionally
small — three markdown notes — so the full pipeline runs in seconds.

## Run it

PowerShell:

```pwsh
./run.ps1
```

bash / zsh:

```bash
./run.sh
```

Either script runs:

```text
llm-patch compile ./notes --output ./out
llm-patch info     ./out
```

## Layout

```
examples/quickstart/
  notes/
    react-hooks.md
    tailwind-tips.md
    pytest-fixtures.md
  run.ps1
  run.sh
```

Drop your own `*.md` files into `notes/` to extend the demo.
