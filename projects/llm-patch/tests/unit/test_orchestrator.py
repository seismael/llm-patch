"""Unit tests for the KnowledgeFusionOrchestrator (Facade)."""

from __future__ import annotations

from unittest.mock import MagicMock

from llm_patch.core.models import AdapterManifest, DocumentContext
from llm_patch.orchestrator import KnowledgeFusionOrchestrator


class TestOrchestratorCallbackRegistration:
    def test_registers_callback_on_source(
        self, mock_source: MagicMock, mock_generator: MagicMock, mock_repository: MagicMock
    ) -> None:
        KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        mock_source.register_callback.assert_called_once()

    def test_registered_callback_is_callable(
        self, mock_source: MagicMock, mock_generator: MagicMock, mock_repository: MagicMock
    ) -> None:
        KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        callback = mock_source.register_callback.call_args[0][0]
        assert callable(callback)


class TestOrchestratorDocumentProcessing:
    def test_on_document_changed_calls_generate_then_save(
        self,
        mock_source: MagicMock,
        mock_generator: MagicMock,
        mock_repository: MagicMock,
        sample_document: DocumentContext,
    ) -> None:
        orchestrator = KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        orchestrator._on_document_changed(sample_document)

        mock_generator.generate.assert_called_once_with(sample_document)
        mock_repository.save.assert_called_once()

        # Verify save was called with correct adapter_id
        save_call = mock_repository.save.call_args
        assert save_call[0][0] == sample_document.document_id

    def test_process_document_returns_manifest(
        self,
        mock_source: MagicMock,
        mock_generator: MagicMock,
        mock_repository: MagicMock,
        sample_document: DocumentContext,
    ) -> None:
        orchestrator = KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        result = orchestrator.process_document(sample_document)

        assert isinstance(result, AdapterManifest)
        assert result.adapter_id == "test_doc"

    def test_process_document_passes_weights_and_config_to_save(
        self,
        mock_source: MagicMock,
        mock_generator: MagicMock,
        mock_repository: MagicMock,
        sample_document: DocumentContext,
    ) -> None:
        orchestrator = KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        orchestrator.process_document(sample_document)

        save_call = mock_repository.save.call_args
        # Positional args: adapter_id, weights, peft_config
        assert save_call[0][0] == sample_document.document_id
        assert save_call[0][1] == mock_generator.generate.return_value
        assert save_call[0][2] == mock_generator.get_peft_config.return_value


class TestOrchestratorCompileAll:
    def test_compile_all_processes_all_documents(
        self, mock_source: MagicMock, mock_generator: MagicMock, mock_repository: MagicMock
    ) -> None:
        orchestrator = KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        manifests = orchestrator.compile_all()

        assert mock_generator.generate.call_count == 3
        assert mock_repository.save.call_count == 3
        assert len(manifests) == 3

    def test_compile_all_returns_list_of_manifests(
        self, mock_source: MagicMock, mock_generator: MagicMock, mock_repository: MagicMock
    ) -> None:
        orchestrator = KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        manifests = orchestrator.compile_all()

        assert all(isinstance(m, AdapterManifest) for m in manifests)


class TestOrchestratorLifecycle:
    def test_start_calls_source_start(
        self, mock_source: MagicMock, mock_generator: MagicMock, mock_repository: MagicMock
    ) -> None:
        orchestrator = KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        orchestrator.start()
        mock_source.start.assert_called_once()

    def test_stop_calls_source_stop(
        self, mock_source: MagicMock, mock_generator: MagicMock, mock_repository: MagicMock
    ) -> None:
        orchestrator = KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        orchestrator.stop()
        mock_source.stop.assert_called_once()

    def test_context_manager(
        self, mock_source: MagicMock, mock_generator: MagicMock, mock_repository: MagicMock
    ) -> None:
        orchestrator = KnowledgeFusionOrchestrator(mock_source, mock_generator, mock_repository)
        with orchestrator:
            mock_source.start.assert_called_once()
        mock_source.stop.assert_called_once()
