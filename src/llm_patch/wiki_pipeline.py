"""WikiPipelineOrchestrator — backward-compat shim.

New code should use ``llm_patch.pipelines.WikiPipeline`` instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from llm_patch.core.config import WikiConfig
from llm_patch.wiki.interfaces import IWikiAgent
from llm_patch.wiki.manager import WikiManager
from llm_patch.wiki.operations import IngestResult, LintReport, QueryResult
from llm_patch.wiki.schema import WikiSchema

logger = logging.getLogger(__name__)


class WikiPipelineOrchestrator:
    """Legacy shim — delegates wiki operations and optional adapter rebuild."""

    def __init__(
        self,
        agent: IWikiAgent,
        config: WikiConfig,
        adapter_orchestrator: object | None = None,
    ) -> None:
        schema: WikiSchema | None = None
        if config.schema_path and config.schema_path.exists():
            schema = WikiSchema.from_file(config.schema_path)

        self._wiki = WikiManager(agent=agent, base_dir=config.base_dir, schema=schema)
        self._adapter_orchestrator = adapter_orchestrator
        self._config = config

    @property
    def wiki(self) -> WikiManager:
        return self._wiki

    def init(self) -> None:
        self._wiki.init()

    def ingest(self, source_path: Path) -> IngestResult:
        result = self._wiki.ingest(source_path)

        if self._adapter_orchestrator is not None:
            self._trigger_adapter_rebuild(result)

        return result

    def query(self, question: str, *, save_as_synthesis: bool = False) -> QueryResult:
        return self._wiki.query(question, save_as_synthesis=save_as_synthesis)

    def lint(self) -> LintReport:
        return self._wiki.lint()

    def compile_all(self) -> list[IngestResult]:
        results = self._wiki.compile_all()

        if self._adapter_orchestrator is not None:
            for result in results:
                self._trigger_adapter_rebuild(result)

        return results

    def status(self) -> dict[str, int]:
        return self._wiki.status()

    def _trigger_adapter_rebuild(self, result: IngestResult) -> None:
        if not hasattr(self._adapter_orchestrator, "compile_all"):
            return

        logger.info(
            "Triggering adapter rebuild for %d touched pages",
            len(result.all_pages_touched),
        )
        try:
            self._adapter_orchestrator.compile_all()  # type: ignore[union-attr]
        except Exception:
            logger.exception("Adapter rebuild failed")
