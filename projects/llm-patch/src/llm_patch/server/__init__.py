"""llm-patch HTTP API server (FastAPI).

Provides REST endpoints for:
- Data source operations (list / preview documents)
- Adapter compilation (compile batch or single document)
- Adapter management (list / info / delete)
- Model inference (generate / chat)
"""

from llm_patch.server.app import app

__all__ = ["app"]
