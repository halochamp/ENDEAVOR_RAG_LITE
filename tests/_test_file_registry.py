# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""_test_file_registry.py — file_registry lifecycle + portability (_rel/_abs roundtrip)"""
import os
import tempfile
from pathlib import Path

from _runner import Runner

r = Runner("file_registry")


def t11_registry():
    from config import KNOWLEDGE_DIR
    from file_registry import check, register, deregister
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w", dir=KNOWLEDGE_DIR
    ) as f:
        f.write("hello")
        path = f.name
    assert check(path) == "new"
    register(path)
    assert check(path) == "skip"
    with open(path, "w") as f:
        f.write("changed content")
    assert check(path) == "changed"
    deregister(path)
    assert check(path) == "new"
    os.unlink(path)


def t30_rel_abs_roundtrip_inside_base():
    """A knowledge-root file round-trips through the portable registry key."""
    from config import KNOWLEDGE_DIR
    from file_registry import _rel, _abs
    inside = KNOWLEDGE_DIR / "_portability_roundtrip_test.tmp"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_text("x")
    try:
        key = _rel(str(inside))
        assert not Path(key).is_absolute(), f"expected relative key, got {key}"
        assert _abs(key) == str(inside.resolve())
    finally:
        inside.unlink()


def t31_rel_falls_back_to_abs_outside_base():
    """Files outside the configured knowledge root are rejected."""
    from file_registry import _rel
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        outside = f.name
    try:
        try:
            _rel(outside)
        except ValueError:
            pass
        else:
            raise AssertionError("outside path was accepted")
    finally:
        os.unlink(outside)


def t46_pipeline_change_invalidates_unchanged_file():
    import embedder
    import file_registry
    from config import KNOWLEDGE_DIR

    saved = (
        file_registry.REGISTRY_PATH,
        file_registry._registry,
        file_registry._loaded,
        file_registry._fingerprint,
        embedder.MODEL_NAME,
    )
    path = None
    try:
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".md", mode="w", dir=KNOWLEDGE_DIR
        ) as f:
            f.write("unchanged")
            path = f.name
        with tempfile.TemporaryDirectory() as tmp:
            file_registry.REGISTRY_PATH = Path(tmp) / "registry.json"
            file_registry._registry = {}
            file_registry._loaded = False
            file_registry._fingerprint = None
            file_registry.register(path)
            assert file_registry.check(path) == "skip"
            embedder.MODEL_NAME += "-new-vector-space"
            assert file_registry.check(path) == "changed"
    finally:
        (
            file_registry.REGISTRY_PATH,
            file_registry._registry,
            file_registry._loaded,
            file_registry._fingerprint,
            embedder.MODEL_NAME,
        ) = saved
        if path:
            Path(path).unlink(missing_ok=True)


r.test("T11 new/skip/changed/deregister", t11_registry)
r.test("T30 knowledge-root _rel/_abs roundtrip", t30_rel_abs_roundtrip_inside_base)
r.test("T31 outside path rejected", t31_rel_falls_back_to_abs_outside_base)
r.test("T46 pipeline/model change invalidates file", t46_pipeline_change_invalidates_unchanged_file)

if __name__ == "__main__":
    r.exit()
