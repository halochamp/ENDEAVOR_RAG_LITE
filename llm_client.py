# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""Local MLX VLM client and project-local server lifecycle."""
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import requests

from config import (
    LOG_DIR as _LOG_DIR,
    MLX_API_KEY,
    MLX_HOST,
    MLX_MODEL,
    MLX_PORT,
    MLX_PREFILL_STEP_SIZE,
    MLX_SERVER_PYTHON,
    NO_AUTO_START,
    ROOT_DIR,
)

MLX_BASE_URL = f"http://{MLX_HOST}:{MLX_PORT}/v1"

# Single-flight guard: without it, concurrent ensure_mlx_server() callers all
# see mlx_server_up()==False and each Popen their own mlx_vlm.server — both
# fully load the model (double RAM, wasted load time) before one loses the
# port bind (Errno 48) and exits. A bare Lock around the check-then-spawn
# decision is NOT enough by itself (model load takes several seconds, so a
# second caller's re-check inside the lock still sees "not up") — must also
# track the in-flight Popen's liveness so concurrent callers cannot load two
# copies of the model before one loses the port bind.
_START_LOCK = threading.Lock()
_START_TIMEOUT_SEC = 120
_start_lock_proc: "subprocess.Popen | None" = None


def mlx_server_up() -> bool:
    try:
        with socket.create_connection((MLX_HOST, MLX_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def ensure_mlx_server(status_cb=None) -> bool:
    """Start ENDEAVOR_RAG's own mlx_vlm.server if it isn't already up.

    This checks/starts only this project's configured local port. It does not
    depend on another project or process being open.
    status_cb(msg: str), if given, is called with short progress strings
    (mirrors ui.Spinner.update_sub in main.py's caller).
    Returns True once the server responds, False on timeout.
    """
    global _start_lock_proc
    if NO_AUTO_START:
        if status_cb:
            status_cb("โหมด manual server: ไม่เปิด mlx_vlm.server อัตโนมัติ")
        return mlx_server_up()
    if mlx_server_up():
        return True

    with _START_LOCK:
        # Re-check: a sibling thread may have finished starting it while this
        # one waited for the lock.
        if mlx_server_up():
            return True

        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / "mlx_server.log"
        already_starting = _start_lock_proc is not None and _start_lock_proc.poll() is None
        if not already_starting:
            if not MLX_SERVER_PYTHON.exists():
                if status_cb:
                    status_cb(f"ไม่พบ mlx_vlm.server python: {MLX_SERVER_PYTHON}")
                return False

            if status_cb:
                status_cb(f"กำลังเปิด mlx_vlm.server ของ ENDEAVOR_RAG ({MLX_MODEL})…")
            with open(log_path, "ab") as log_fh:
                _start_lock_proc = subprocess.Popen(
                    [
                        str(MLX_SERVER_PYTHON), "-m", "mlx_vlm.server",
                        "--model", MLX_MODEL,
                        "--host", MLX_HOST,
                        "--port", str(MLX_PORT),
                        "--api-key", MLX_API_KEY,
                        "--prefill-step-size", str(MLX_PREFILL_STEP_SIZE),
                    ],
                    cwd=str(ROOT_DIR),
                    stdout=log_fh, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

    deadline = time.monotonic() + _START_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if mlx_server_up():
            if status_cb:
                status_cb("mlx_vlm.server พร้อมใช้งาน ✓")
            return True
        time.sleep(1.0)
    if status_cb:
        status_cb(f"mlx_vlm.server ไม่ขึ้นภายใน {_START_TIMEOUT_SEC}s — ดู {log_path}")
    return False


def chat(prompt: str, *, temperature: float = 0.0, max_tokens: int = 64,
         timeout: float = 30.0) -> str:
    """One non-streaming completion. Raises on failure — every caller here
    already wraps this in its own try/except with a conservative fallback,
    matching the old ollama.chat() call sites' error-handling shape."""
    resp = requests.post(
        f"{MLX_BASE_URL}/chat/completions",
        json={
            "model": MLX_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        # mlx_vlm.server 0.6.8 enforces --api-key even when it's the "x"
        # placeholder (older versions didn't) — confirmed live 2026-07-29
        # after upgrading this Mac's `mlx` env for OptiQ support, so always
        # send it rather than skipping on "x" (matches build_llm()'s
        # ChatOpenAI below, which already always sends api_key regardless).
        headers={"Authorization": f"Bearer {MLX_API_KEY}"} if MLX_API_KEY else {},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def build_llm(**overrides):
    """LangChain-compatible client for the local OpenAI-shaped endpoint."""
    from langchain_openai import ChatOpenAI

    params = dict(
        base_url=MLX_BASE_URL,
        api_key=MLX_API_KEY,
        model=MLX_MODEL,
        temperature=0.1,
        streaming=False,
        timeout=httpx.Timeout(10.0, read=90.0, write=10.0, pool=10.0),
    )
    params.update(overrides)
    return ChatOpenAI(**params)
