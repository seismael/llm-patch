"""Wiki pipeline — wiki ingestion → adapter compilation closed loop."""

from __future__ import annotations

import logging
from pathlib import Path

from llm_patch.core.config import WikiConfig
from llm_patch.wiki.interfaces import IWikiAgent
from llm_patch.wiki.manager import WikiManager
from llm_patch.wiki.operations import IngestResult, LintReport, QueryResult
from llm_patch.wiki.schema import WikiSchema

logger = logging.getLogger(__name__)


class WikiPipeline:
    """Composes WikiManager with an optional compile pipeline.

    Owns the wiki lifecycle: init, ingest, query, lint, compile.
    When a ``compile_pipeline`` is provided, wiki page changes are
    forwarded to the adapter compilation pipeline automatically.

    Args:
        agent: The LLM agent driving wiki operations.
        config: WikiConfig with base_dir, schema_path, etc.
        compile_pipeline: Optional ``CompilePipeline`` to compile adapters
            from wiki pages.
    """

    def __init__(
        self,
        agent: IWikiAgent,
        config: WikiConfig,
        compile_pipeline: object | None = None,
    ) -> None:
        schema: WikiSchema | None = None
        if config.schema_path and config.schema_path.exists():
            schema = WikiSchema.from_file(config.schema_path)

        self._wiki = WikiManager(agent=agent, base_dir=config.base_dir, schema=schema)
        self._compile_pipeline = compile_pipeline
        self._config = config

    @property
    def wiki(self) -> WikiManager:
        return self._wiki

    def init(self) -> None:
        self._wiki.init()

    def ingest(self, source_path: Path) -> IngestResult:
        result = self._wiki.ingest(source_path)

        if self._compile_pipeline is not None:
            self._trigger_compile(result)

        return result

    def query(self, question: str, *, save_as_synthesis: bool = False) -> QueryResult:
        return self._wiki.query(question, save_as_synthesis=save_as_synthesis)

    def lint(self) -> LintReport:
        return self._wiki.lint()

    def compile_all(self) -> list[IngestResult]:
        results = self._wiki.compile_all()

        if self._compile_pipeline is not None:
            for result in results:
                self._trigger_compile(result)

        return results

    def status(self) -> dict[str, int]:
        return self._wiki.status()

    def _trigger_compile(self, result: IngestResult) -> None:
        if not hasattr(self._compile_pipeline, "compile_all"):
            return

        logger.info(
            "Triggering adapter rebuild for %d touched pages",
            len(result.all_pages_touched),
        )
        try:
            self._compile_pipeline.compile_all()  # type: ignore[union-attr]
        except Exception:
            logger.exception("Adapter rebuild failed")
