"""llm_patch_wiki_agent — wiki-specialized agent built on llm-patch.

This package consumes the **public** ``llm_patch`` API only (re-exports
from ``llm_patch.__init__``). It must not import from internal engine
modules. See ``AGENTS.md`` for the per-project contract and the root
``SPEC.md`` for the dependency-direction rule.
"""

__version__ = "0.1.0"

from llm_patch_wiki_agent.agent import WikiAgent, WikiAgentConfig, WikiAgentInfo

__all__ = ["WikiAgent", "WikiAgentConfig", "WikiAgentInfo", "__version__"]
