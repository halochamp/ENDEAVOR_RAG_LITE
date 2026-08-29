# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""_test_e2e.py — full ingest -> search -> delete round trip"""
import os
import tempfile
from pathlib import Path

from _runner import Runner

r = Runner("e2e")


def t28_e2e():
    from ingestor import ingest_file, delete_file
    from retriever import search, fetch_parents

    content = ("Retrieval Augmented Generation (RAG) combines LLMs with external knowledge.\n"
               "It retrieves relevant documents before generating an answer.\n\n") * 8

    from config import KNOWLEDGE_DIR
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w", dir=KNOWLEDGE_DIR
    ) as f:
        f.write(content)
        path = Path(f.name)

    try:
        n = ingest_file(path)
        assert n > 0, "No chunks stored"

        chunks = search(["RAG retrieval augmented generation"], top_k=5, top_fused=3)
        parents = fetch_parents(chunks)
        assert len(parents) > 0, "No results found after ingest"
        assert any("RAG" in p["parent_text"] or "retrieval" in p["parent_text"].lower()
                   for p in parents)

        delete_file(path)
    finally:
        os.unlink(path)


def t40_filename_containing_chroma_is_indexable():
    import ingestor

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chromatic.md"
        path.write_text("supported document")
        assert ingestor._is_indexable_file(path)


def t41_rejected_symlink_does_not_abort_build():
    import main
    from config import KNOWLEDGE_DIR

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        outside = Path(tmp) / "outside.txt"
        outside.write_text("outside document")
        link = KNOWLEDGE_DIR / "escape.txt"
        link.symlink_to(outside)
        try:
            main._run_build()
        finally:
            link.unlink(missing_ok=True)


def t55_unregistered_partial_source_is_rebuilt():
    import file_registry
    import ingestor
    import store
    from config import KNOWLEDGE_DIR

    saved = {
        "DATA_DIR": ingestor.DATA_DIR,
        "check": file_registry.check,
        "all_registered": file_registry.all_registered,
        "has_source": store.has_source,
        "delete_file": ingestor.delete_file,
        "deregister": file_registry.deregister,
        "ingest_file": ingestor.ingest_file,
        "register": file_registry.register,
        "health_check": store.health_check,
        "ghost_files": file_registry.ghost_files,
    }
    deleted = []
    path = None
    try:
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".md", mode="w", dir=KNOWLEDGE_DIR
        ) as f:
            f.write("supported document")
            path = Path(f.name)
        ingestor.DATA_DIR = KNOWLEDGE_DIR
        file_registry.check = lambda value: "new"
        file_registry.all_registered = lambda: []
        store.has_source = lambda source: True
        ingestor.delete_file = lambda value: deleted.append(Path(value).name) or 1
        file_registry.deregister = lambda value: None
        ingestor.ingest_file = lambda value: 1
        file_registry.register = lambda value: None
        store.health_check = lambda: []
        file_registry.ghost_files = lambda: []
        ingestor.sync_knowledge_base()
        assert deleted == [path.name], deleted
    finally:
        if path:
            path.unlink(missing_ok=True)
        ingestor.DATA_DIR = saved["DATA_DIR"]
        file_registry.check = saved["check"]
        file_registry.all_registered = saved["all_registered"]
        store.has_source = saved["has_source"]
        ingestor.delete_file = saved["delete_file"]
        file_registry.deregister = saved["deregister"]
        ingestor.ingest_file = saved["ingest_file"]
        file_registry.register = saved["register"]
        store.health_check = saved["health_check"]
        file_registry.ghost_files = saved["ghost_files"]


r.test("T28 ingest txt + search e2e", t28_e2e)
r.test("T40 supported filename containing chroma remains indexable", t40_filename_containing_chroma_is_indexable)
r.test("T41 rejected symlink does not abort build", t41_rejected_symlink_does_not_abort_build)
r.test("T55 unregistered partial source is rebuilt", t55_unregistered_partial_source_is_rebuilt)

if __name__ == "__main__":
    r.exit()
