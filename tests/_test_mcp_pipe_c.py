"""Deterministic Pipe C (MCP) -> Pipe B contract tests; no live model required."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from unittest import mock

from _runner import Runner

r = Runner("mcp pipe c")


def _server_module():
    import mcp_server

    return mcp_server


def t_catalog_is_only_pipe_b_retrieval():
    server = _server_module()
    tools = asyncio.run(server.mcp.list_tools())
    assert [item.name for item in tools] == ["rag_retrieve"]
    schema = tools[0].inputSchema
    assert set(schema["properties"]) == {
        "query",
        "mode",
        "tags",
        "filename_contains",
        "created_after",
        "created_before",
        "source_type",
    }
    assert schema["required"] == ["query"]


def t_pipe_c_does_not_load_pipe_a_or_llm():
    server = _server_module()
    assert server is not None
    assert "main" not in sys.modules
    assert "llm_client" not in sys.modules
    assert "rag_search" not in sys.modules


def t_pipe_b_module_resolves_inside_this_rag_tree():
    _server_module()
    import rag_retrieve

    expected_root = Path(__file__).resolve().parents[1]
    assert Path(rag_retrieve.__file__).resolve().parent == expected_root


def t_delegates_validated_request_to_pipe_b():
    server = _server_module()
    captured = {}

    def fake_invoke(arguments):
        captured.update(arguments)
        return "PIPE_B_RESULT"

    with mock.patch.object(server, "_invoke_pipe_b", side_effect=fake_invoke):
        result = server.rag_retrieve(
            query=[" Thai query ", "English query"],
            mode=" SOURCE_FIRST ",
            tags=" finance ",
            filename_contains=" report ",
            created_after="2026-01-01",
            created_before="2026-12-31",
            source_type=" md ",
        )

    assert result == "PIPE_B_RESULT"
    assert captured == {
        "query": ["Thai query", "English query"],
        "mode": "source_first",
        "tags": "finance",
        "filename_contains": "report",
        "created_after": "2026-01-01",
        "created_before": "2026-12-31",
        "source_type": "md",
    }


def t_rejects_invalid_requests_before_pipe_b():
    server = _server_module()
    invalid = [
        {"query": ""},
        {"query": "x" * (server._MAX_QUERY_CHARS + 1)},
        {"query": []},
        {"query": ["ok"] * 9},
        {"query": "ok", "mode": "unknown"},
        {"query": "ok", "created_after": "2026-02-30"},
        {"query": "ok", "created_after": "2026-12-31", "created_before": "2026-01-01"},
        {"query": "ok", "tags": "x" * (server._MAX_FILTER_CHARS + 1)},
        {"query": "ok", "tags": 123},
        {"query": "ok", "filename_contains": []},
        {"query": "ok", "created_before": None},
        {"query": 123},
    ]
    with mock.patch.object(server, "_invoke_pipe_b") as invoke:
        for arguments in invalid:
            try:
                server.rag_retrieve(**arguments)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid request: {arguments}")
        invoke.assert_not_called()


def t_accepts_documented_boundary_lengths():
    server = _server_module()
    with mock.patch.object(server, "_invoke_pipe_b", return_value="boundary-ok") as invoke:
        result = server.rag_retrieve(
            query="x" * server._MAX_QUERY_CHARS,
            tags="x" * server._MAX_FILTER_CHARS,
        )
    assert result == "boundary-ok"
    invoke.assert_called_once()


def t_pipe_b_error_becomes_mcp_execution_error():
    server = _server_module()
    with mock.patch.object(server, "_invoke_pipe_b", return_value="[error] index unavailable"):
        try:
            server.rag_retrieve("query")
        except RuntimeError as exc:
            assert str(exc) == "Pipe B rag_retrieve returned an error"
        else:
            raise AssertionError("Pipe B error was returned as a successful result")


def t_pipe_b_exception_is_mapped_without_leaking_details():
    server = _server_module()
    secret_detail = "secret-path-/tmp/private"
    with mock.patch.object(server, "_invoke_pipe_b", side_effect=ValueError(secret_detail)):
        try:
            server.rag_retrieve("query")
        except RuntimeError as exc:
            assert "ValueError" in str(exc)
            assert secret_detail not in str(exc)
            assert exc.__cause__ is None
        else:
            raise AssertionError("Pipe B exception was returned as a successful result")


def t_non_text_pipe_b_result_is_rejected():
    server = _server_module()
    with mock.patch.object(server, "_invoke_pipe_b", return_value={"result": "not text"}):
        try:
            server.rag_retrieve("query")
        except RuntimeError as exc:
            assert str(exc) == "Pipe B rag_retrieve returned a non-text result"
        else:
            raise AssertionError("non-text Pipe B result was returned as success")


def t_oversized_result_is_explicitly_capped():
    server = _server_module()
    with mock.patch.object(server, "_MAX_OUTPUT_CHARS", 256), mock.patch.object(
        server, "_invoke_pipe_b", return_value="x" * 1_000
    ):
        result = server.rag_retrieve("query")
    assert len(result) <= 256
    assert result.startswith("x")
    assert "[truncated]" in result


def t_tiny_cap_never_exceeds_configured_limit():
    server = _server_module()
    with mock.patch.object(server, "_MAX_OUTPUT_CHARS", 32), mock.patch.object(
        server, "_invoke_pipe_b", return_value="x" * 1_000
    ):
        result = server.rag_retrieve("query")
    assert len(result) <= 32
    assert "[truncated]" in result


def t_result_at_or_below_cap_is_unchanged():
    server = _server_module()
    expected = "stable result"
    with mock.patch.object(server, "_MAX_OUTPUT_CHARS", len(expected)), mock.patch.object(
        server, "_invoke_pipe_b", return_value=expected
    ):
        assert server.rag_retrieve("query") == expected


def t_pipe_b_calls_are_serialized():
    server = _server_module()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    entered: list[int] = []

    def fake_invoke(arguments):
        call_no = len(entered) + 1
        entered.append(call_no)
        if call_no == 1:
            first_entered.set()
            assert not second_entered.is_set()
            assert release_first.wait(2), "first Pipe B call did not receive release"
        else:
            second_entered.set()
        return f"ok-{call_no}"

    results: list[str] = []

    def call():
        results.append(server.rag_retrieve("query"))

    with mock.patch.object(server, "_invoke_pipe_b", side_effect=fake_invoke):
        first = threading.Thread(target=call)
        second = threading.Thread(target=call)
        try:
            first.start()
            assert first_entered.wait(2), "first Pipe B call did not start"
            second.start()
            assert not second_entered.wait(0.1), "second Pipe B call entered before first left"
        finally:
            release_first.set()
            first.join(2)
            second.join(2)

    assert not first.is_alive() and not second.is_alive()
    assert sorted(results) == ["ok-1", "ok-2"]


def t_stdio_handshake_lists_pipe_c_without_starting_a_model():
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server_path = Path(__file__).resolve().parents[1] / "mcp_server.py"

    async def handshake():
        parameters = StdioServerParameters(command=sys.executable, args=[str(server_path)])
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                return initialized.serverInfo.name, [item.name for item in listed.tools]

    name, tools = asyncio.run(handshake())
    assert name == "ENDEAVOR_RAG Pipe C"
    assert tools == ["rag_retrieve"]


r.test("catalog exposes only rag_retrieve", t_catalog_is_only_pipe_b_retrieval)
r.test("Pipe C does not load Pipe A or its LLM", t_pipe_c_does_not_load_pipe_a_or_llm)
r.test("Pipe B resolves inside this RAG tree", t_pipe_b_module_resolves_inside_this_rag_tree)
r.test("valid request delegates to Pipe B", t_delegates_validated_request_to_pipe_b)
r.test("invalid request is rejected before Pipe B", t_rejects_invalid_requests_before_pipe_b)
r.test("documented boundary lengths are accepted", t_accepts_documented_boundary_lengths)
r.test("Pipe B error becomes MCP execution error", t_pipe_b_error_becomes_mcp_execution_error)
r.test("Pipe B exception details are not leaked", t_pipe_b_exception_is_mapped_without_leaking_details)
r.test("non-text Pipe B result is rejected", t_non_text_pipe_b_result_is_rejected)
r.test("oversized output is explicitly capped", t_oversized_result_is_explicitly_capped)
r.test("tiny output cap never overflows", t_tiny_cap_never_exceeds_configured_limit)
r.test("output at the cap is unchanged", t_result_at_or_below_cap_is_unchanged)
r.test("Pipe B calls are serialized", t_pipe_b_calls_are_serialized)
r.test("stdio handshake lists Pipe C without starting a model", t_stdio_handshake_lists_pipe_c_without_starting_a_model)

if __name__ == "__main__":
    r.exit()
