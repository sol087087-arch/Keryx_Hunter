"""Smoke tests: every core module must import cleanly."""
import importlib
import pytest


MODULES = [
    "keryx.models.interface",
    "keryx.models.local_mock",
    "keryx.models.remote",
    "keryx.models.swarm",
    "keryx.core.shared_context",
    "keryx.core.agent",
    "keryx.core.orchestrator",
    "keryx.core.router",
    "keryx.advisors.base",
    "keryx.advisors.manager",
    "keryx.advisors.cascade_advisor",
    "keryx.advisors.local_advisor",
    "keryx.advisors.cloud_advisor",
    "keryx.tools.Toolbox",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
