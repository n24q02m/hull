import importlib
import importlib.metadata
from unittest.mock import patch

import hull_embedding_daemon


def test_version_exposed() -> None:
    assert isinstance(hull_embedding_daemon.__version__, str)
    assert len(hull_embedding_daemon.__version__) > 0


def test_version_comes_from_hull_core_dist() -> None:
    """The daemon ships inside the hull-core wheel; there is no separate dist."""
    assert hull_embedding_daemon.__version__ == importlib.metadata.version("hull-core")


def test_version_fallback() -> None:
    """Test that __version__ falls back when package is not installed."""
    with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
        # We need to reload the module to re-run the try/except block
        importlib.reload(hull_embedding_daemon)
        assert hull_embedding_daemon.__version__ == "0.0.0+unknown"

    # Restore the real version for other tests
    importlib.reload(hull_embedding_daemon)
