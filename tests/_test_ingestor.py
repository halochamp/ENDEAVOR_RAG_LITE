# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""_test_ingestor.py — ingestor file loaders + portability (_rel out-of-tree fallback)"""
import csv
import json
import os
import tempfile
from pathlib import Path

from _runner import Runner

r = Runner("ingestor")


def t12_txt():
    from ingestor import load_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as f:
        f.write("hello world machine learning")
        path = Path(f.name)
    text = load_file(path)
    os.unlink(path)
    assert "machine learning" in text


def t13_md():
    from ingestor import load_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w") as f:
        f.write("# Title\n\ncontent here")
        path = Path(f.name)
    text = load_file(path)
    os.unlink(path)
    assert "content" in text


def t14_csv():
    from ingestor import load_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as f:
        csv.writer(f).writerows([["name", "age"], ["Alice", "30"], ["Bob", "25"]])
        path = Path(f.name)
    text = load_file(path)
    os.unlink(path)
    assert "[columns: name, age]" in text
    assert "Alice" in text


def t15_json():
    from ingestor import load_file
    data = {"users": [{"name": "Alice", "age": 30}], "version": 1}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        json.dump(data, f)
        path = Path(f.name)
    text = load_file(path)
    os.unlink(path)
    assert "[json schema:" in text
    assert "Alice" in text
    assert "version" in text


def t16_csv_header_every_chunk():
    from ingestor import _load_csv
    from chunker import chunk_document
    rows = [["col1", "col2", "col3"]] + [[f"val{i}a", f"val{i}b", f"val{i}c"] for i in range(20)]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as f:
        csv.writer(f).writerows(rows)
        path = Path(f.name)
    text = _load_csv(path)
    os.unlink(path)
    chunks = chunk_document(text)
    for c in chunks:
        assert "[columns:" in c["child_text"], f"Missing header in chunk: {c['child_text'][:100]}"


def t17_json_header_every_chunk():
    from ingestor import _load_json
    from chunker import chunk_document
    data = {"items": [{"id": i, "val": f"value_{i}" * 5} for i in range(30)]}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        json.dump(data, f)
        path = Path(f.name)
    text = _load_json(path)
    os.unlink(path)
    chunks = chunk_document(text)
    for c in chunks:
        assert "[json schema:" in c["child_text"], f"Missing header in chunk: {c['child_text'][:100]}"


def t32_rel_falls_back_to_abs_outside_base():
    """ingestor._rel() rejects files outside the configured knowledge root."""
    from ingestor import _rel
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        outside = Path(f.name)
    try:
        try:
            _rel(outside)
        except ValueError:
            pass
        else:
            raise AssertionError("outside path was accepted")
    finally:
        os.unlink(outside)


def t33_rel_inside_base_is_relative():
    """ingestor._rel() returns a relative key inside the knowledge root."""
    from config import KNOWLEDGE_DIR
    from ingestor import _rel
    inside = KNOWLEDGE_DIR / "_portability_ingestor_test.tmp"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_text("x")
    try:
        key = _rel(inside)
        assert not Path(key).is_absolute(), f"expected relative key, got {key}"
        assert key == inside.name
    finally:
        inside.unlink()


def t53_novelty_check_fails_open_on_llm_outage():
    import ingestor
    original = ingestor._llm_chat
    try:
        ingestor._llm_chat = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("probe"))
        assert ingestor._llm_novelty("new candidate", "old neighbour") is True
    finally:
        ingestor._llm_chat = original


def t54_partial_dual_store_write_is_rolled_back():
    import ingestor
    from config import KNOWLEDGE_DIR

    saved = {
        "encode": ingestor.embedder.encode,
        "should_store": ingestor._should_store,
        "upsert": ingestor.store.upsert_chunk,
        "bm25_add": ingestor.store.bm25_add,
        "bm25_flush": ingestor.store.bm25_flush,
        "delete": ingestor.store.delete_by_source,
        "bm25_delete": ingestor.store.bm25_delete_by_source,
    }
    calls = []
    path = None
    try:
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", dir=KNOWLEDGE_DIR
        ) as f:
            f.write("rollback probe content")
            path = Path(f.name)
        ingestor.embedder.encode = lambda texts: [[1.0, 0.0] for _ in texts]
        ingestor._should_store = lambda text, vec: True
        ingestor.store.upsert_chunk = lambda **kwargs: "chunk-id"
        ingestor.store.bm25_add = lambda *args, **kwargs: None
        ingestor.store.bm25_flush = lambda: (_ for _ in ()).throw(OSError("disk full"))
        ingestor.store.delete_by_source = lambda source: calls.append(("dense", source)) or 1
        ingestor.store.bm25_delete_by_source = lambda source: calls.append(("bm25", source))
        try:
            ingestor.ingest_file(path)
        except OSError:
            pass
        else:
            raise AssertionError("expected controlled flush failure")
        assert [kind for kind, _ in calls] == ["dense", "bm25"], calls
    finally:
        if path:
            path.unlink(missing_ok=True)
        ingestor.embedder.encode = saved["encode"]
        ingestor._should_store = saved["should_store"]
        ingestor.store.upsert_chunk = saved["upsert"]
        ingestor.store.bm25_add = saved["bm25_add"]
        ingestor.store.bm25_flush = saved["bm25_flush"]
        ingestor.store.delete_by_source = saved["delete"]
        ingestor.store.bm25_delete_by_source = saved["bm25_delete"]


r.test("T12 txt loader", t12_txt)
r.test("T13 md loader", t13_md)
r.test("T14 csv loader + header", t14_csv)
r.test("T15 json loader + schema", t15_json)
r.test("T16 csv header in every child chunk", t16_csv_header_every_chunk)
r.test("T17 json schema in every child chunk", t17_json_header_every_chunk)
r.test("T32 outside path rejected", t32_rel_falls_back_to_abs_outside_base)
r.test("T33 knowledge-root _rel is relative", t33_rel_inside_base_is_relative)
r.test("T53 novelty outage preserves candidate", t53_novelty_check_fails_open_on_llm_outage)
r.test("T54 partial Chroma/BM25 write rolls back", t54_partial_dual_store_write_is_rolled_back)

if __name__ == "__main__":
    r.exit()
