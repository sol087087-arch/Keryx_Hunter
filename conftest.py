"""
conftest.py — pytest path bootstrap.

The project root contains a __init__.py (public API shim) which causes pytest
to treat the Keryx_Hunter directory itself as a package and resolve imports as
`Keryx_Hunter.keryx.xxx` instead of `keryx.xxx`.  Inserting the project root
at the front of sys.path before collection starts restores the correct layout.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
