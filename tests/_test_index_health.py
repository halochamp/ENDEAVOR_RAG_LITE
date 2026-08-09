# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""Whole-KB consistency checks against an isolated temporary index."""
from _runner import Runner

r = Runner("index_health")


def t36_index_health_check():
    from store import health_check
    issues = health_check()
    assert not issues, "; ".join(issues)


def t38_registry_files_have_chunks():
    """Informational: registry entries with zero chunks (benign dedup ghosts) shouldn't silently grow unbounded."""
    from file_registry import ghost_files
    ghosts = ghost_files()
    assert len(ghosts) <= 10, f"{len(ghosts)} registered files have zero chunks (expected <=10 known dedup ghosts): {ghosts}"


def t39_ghost_files_ignores_malformed_metadata():
    import file_registry
    import store

    class FakeCollection:
        def count(self):
            return 1

        def get(self, include=None):
            return {"metadatas": [{}]}

    original_collection = store._get_collection
    original_registry = file_registry.all_registered
    try:
        store._get_collection = lambda: FakeCollection()
        file_registry.all_registered = lambda: []
        assert file_registry.ghost_files() == []
    finally:
        store._get_collection = original_collection
        file_registry.all_registered = original_registry


r.test("T36 store.health_check() — bm25/chroma sync", t36_index_health_check)
r.test("T38 registry ghost-file count bounded", t38_registry_files_have_chunks)
r.test("T39 malformed Chroma metadata does not crash ghost check", t39_ghost_files_ignores_malformed_metadata)

if __name__ == "__main__":
    r.exit()
