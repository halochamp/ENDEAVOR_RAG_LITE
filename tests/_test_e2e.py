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
    import main

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chromatic.md"
        path.write_text("supported document")
        assert main._is_indexable_file(path)


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
    import main
    from config import KNOWLEDGE_DIR

    saved = {
        "DATA_DIR": main.DATA_DIR,
        "check": main.check,
        "all_registered": main.all_registered,
        "has_source": main.store.has_source,
        "delete_file": main.delete_file,
        "deregister": main.deregister,
        "ingest_file": main.ingest_file,
        "register": main.register,
        "health_check": main.store.health_check,
        "ghost_files": main.ghost_files,
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
        main.DATA_DIR = KNOWLEDGE_DIR
        main.check = lambda value: "new"
        main.all_registered = lambda: []
        main.store.has_source = lambda source: True
        main.delete_file = lambda value: deleted.append(Path(value).name) or 1
        main.deregister = lambda value: None
        main.ingest_file = lambda value: 1
        main.register = lambda value: None
        main.store.health_check = lambda: []
        main.ghost_files = lambda: []
        main._run_build()
        assert deleted == [path.name], deleted
    finally:
        if path:
            path.unlink(missing_ok=True)
        for name, value in saved.items():
            setattr(main, name, value)


r.test("T28 ingest txt + search e2e", t28_e2e)
r.test("T40 supported filename containing chroma remains indexable", t40_filename_containing_chroma_is_indexable)
r.test("T41 rejected symlink does not abort build", t41_rejected_symlink_does_not_abort_build)
r.test("T55 unregistered partial source is rebuilt", t55_unregistered_partial_source_is_rebuilt)

if __name__ == "__main__":
    r.exit()
