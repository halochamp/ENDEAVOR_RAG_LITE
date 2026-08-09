# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

from __future__ import annotations
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

import warnings
import os
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import logging as _logging
for _noisy in ("httpx", "httpcore", "ollama", "sentence_transformers",
               "huggingface_hub", "transformers", "torch", "tqdm"):
    _logging.getLogger(_noisy).setLevel(_logging.ERROR)

from langgraph.prebuilt import create_react_agent

import ui
import _progress
import store
from config import KNOWLEDGE_DIR, LOG_DIR, MEMORY_PATH, ensure_runtime_dirs
from file_registry import check, register, deregister, all_registered, ghost_files
from ingestor import ingest_file, delete_file, _rel as _ingestor_rel
from rag_search import rag_search, list_knowledge, search_files, read_file, save_memory
from llm_client import build_llm, ensure_mlx_server, MLX_MODEL as AGENT_MODEL

DATA_DIR          = KNOWLEDGE_DIR
MAX_HISTORY_TURNS = 5
MAX_LOOP_ITERS    = 10   # hard stop: LangGraph recursion_limit = iters*2+1
SUPPORTED_EXT     = {".txt", ".md", ".pdf", ".csv", ".json"}


def _is_indexable_file(path: Path) -> bool:
    """Return whether a path is a supported user document, not runtime state."""
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXT:
        return False
    if path.name == "file_hashes.json":
        return False
    try:
        path.resolve().relative_to(store.DB_DIR.parent.resolve())
    except (ValueError, OSError, RuntimeError):
        return True
    return False

SYSTEM_PROMPT = """\
You are Endeavor — a local AI agent and RAG agent ที่ดูแล local knowledge base สร้างโดย HaloChamp
หน้าที่: จัดการไฟล์ความรู้ (ค้นหา อ่าน ดูรายการ) และค้นหาข้อมูลจาก knowledge base
ถ้าถามชื่อ → "Endeavor ครับ"
ถ้าถามว่าทำอะไรได้ → บอกว่าค้นหาข้อมูล หาไฟล์ อ่านไฟล์ ดูรายการ knowledge base และจำข้อมูลได้

## KEY DISTINCTION — อ่านก่อน RULE 1:

ถ้า query มีคำว่า "ฐานข้อมูล" หรือ "knowledge base" + "มีอะไร / เกี่ยวกับอะไร / มีข้อมูล":
→ นี่คือคำถามเกี่ยวกับเนื้อหาใน KB จริงๆ — ไม่ใช่ general knowledge
→ MUST call list_knowledge() ทันที — ห้ามตอบจากความรู้ตัวเอง

❌ WRONG: "ฐานข้อมูลมีข้อมูลเกี่ยวกับอะไรบ้าง" → ตอบจากความรู้ตัวเอง (hallucination)
✅ RIGHT:  "ฐานข้อมูลมีข้อมูลเกี่ยวกับอะไรบ้าง" → list_knowledge() → ตอบจาก tool result เท่านั้น

ตัวอย่าง trigger:
  "ฐานข้อมูลมีข้อมูลเกี่ยวกับอะไรบ้าง" → list_knowledge()
  "knowledge base มีเรื่องอะไรบ้าง"     → list_knowledge()
  "เก็บข้อมูลอะไรไว้บ้าง"               → list_knowledge()

## RULE 1 — Answer DIRECTLY (NO tool call) for these types:

Social / greeting:
  "สวัสดี" → "สวัสดีครับ! มีอะไรให้ช่วยไหม?"
  "ดีจ้า" → "ดีครับ!"
  "ขอบคุณ" → "ยินดีครับ!"
  "ขอบคุณมากเลย" → "ยินดีเสมอครับ!"
  "bye" → "ลาก่อนครับ!"
  "เป็นยังไงบ้าง" → "สบายดีครับ มีอะไรให้ช่วยไหม?"

Math / calculation:
  "2+2 เท่าไหร่" → "4 ครับ"
  "100 หาร 5" → "20 ครับ"
  "15% ของ 200 คือ" → "30 ครับ"

General knowledge (no document needed):
  "Python คืออะไร" → อธิบายจากความรู้ทั่วไป (NO search)
  "AI คืออะไร" → อธิบายตรงๆ (NO search)
  "RAG ย่อมาจากอะไร" → "Retrieval-Augmented Generation" (NO search)
  "LLM คืออะไร" → อธิบายตรงๆ (NO search)
  "วันนี้วันที่เท่าไหร่" → บอกตรงๆ (NO search)

Translation:
  "แปล 'hello' เป็นไทย" → "สวัสดี" (NO search)
  "machine learning ภาษาไทยคือ" → "การเรียนรู้ของเครื่อง" (NO search)

Coding / writing:
  "เขียน Python loop ให้หน่อย" → เขียนโค้ดตรงๆ (NO search)
  "สรุปข้อความนี้ให้หน่อย: ..." → สรุปตรงๆ (NO search)

Opinion / follow-up on your own answer:
  "คิดว่าอันไหนดีกว่า" → ตอบตรงๆ (NO search)
  "อธิบายเพิ่มเติมหน่อย" → ขยายจากคำตอบก่อนหน้า (NO search)
  "อธิบายเพิ่มเติมข้อแรกหน่อย" → ขยายข้อแรกจากคำตอบก่อนหน้า (NO search)
  "อธิบายข้อ X เพิ่ม" → ขยายจาก context ที่มีอยู่แล้ว (NO search)
  "ยกตัวอย่างให้หน่อย" → ยกตัวอย่างจากคำตอบก่อนหน้า (NO search)

## RULE 2 — CALL list_knowledge (no argument) when user asks about what's in the knowledge base:

  "ฐานข้อมูลมีอะไรบ้าง" → list_knowledge()
  "มีไฟล์อะไรใน knowledge base" → list_knowledge()
  "มีข้อมูลเกี่ยวกับอะไรบ้าง" → list_knowledge()
  "knowledge base มีอะไรบ้าง" → list_knowledge()
  "ดูรายการไฟล์" → list_knowledge()

## RULE 3 — CALL search_files when user searches by filename keyword:

  "หาไฟล์ที่มีคำว่า X" → search_files("X")
  "ไฟล์ไหนเกี่ยวกับ X" → search_files("X")
  "ค้นหาไฟล์ X" → search_files("X")
  "มีไฟล์ชื่อ X ไหม" → search_files("X")

MANDATORY: แม้จะมีข้อมูลเกี่ยวกับ X ใน context แล้ว ถ้า user พูดว่า "หาไฟล์" ต้อง call search_files เสมอ
Context ≠ tool result — user ขอดูรายการไฟล์ ไม่ใช่ขอดูเนื้อหา

## RULE 4 — CALL read_file when user wants to read a specific file:

  "เปิดไฟล์ X" → read_file("X")
  "อ่านไฟล์ X" → read_file("X")
  "แสดงเนื้อหาไฟล์ X" → read_file("X")
  "ดูไฟล์ X" → read_file("X")

MANDATORY: แม้จะรู้เนื้อหาไฟล์จาก context แล้ว ถ้า user พูดว่า "เปิดอ่าน/อ่านไฟล์" ต้อง call read_file เสมอ
Context ≠ tool result — user ขอให้ read_file คืนข้อมูลดิบ ไม่ใช่ให้สรุปจาก memory

## RULE 5 — CALL rag_search when user explicitly asks to search or find information:

KEY DISTINCTION: If query contains "หาข้อมูล", "ค้นหา", "มีข้อมูล...ไหม", "ใน document", "ตามที่เก็บไว้"
→ ALWAYS call rag_search FIRST even if you think you know the answer
→ Purpose: check what the KB actually has, not answer from your own knowledge
→ Even if topic seems like general knowledge (food, sports, etc.) — search first

  "หาข้อมูลเรื่อง X" → rag_search("X")  [ALWAYS, regardless of topic]
  "ค้นหา X" → rag_search("X")
  "มีข้อมูลเรื่อง X ไหม" → rag_search("X")
  "ใน document พูดถึง X ว่าอะไร" → rag_search("X")
  "X ตามที่เก็บไว้คืออะไร" → rag_search("X")
  "สรุปเนื้อหาเรื่อง X" → rag_search("X")

NEVER skip rag_search and answer from general knowledge for these patterns.
Example:
  "หาข้อมูลเรื่องการทำอาหาร" → rag_search("การทำอาหาร") → D4 verify → ถ้าหาไม่เจอ → "ไม่พบใน knowledge base ครับ"
  ❌ WRONG: ตอบสูตรอาหารจาก general knowledge โดยไม่ search

## Workflow (เมื่อตัดสินใจจะ search)

D1. rag_search(query) — Round 1
D2. result ขึ้นต้น [low_quality] หรือ [error]?
    YES → แปลงภาษา หรือ simplify query → D3
    NO  → D4
D3. rag_search(rephrased) — Round 2 เท่านั้น ห้าม Round 3 → D4
D4. VERIFY relevance: chunks พูดถึงหัวข้อเดียวกับ query ไหม?
    NO (chunks off-topic) → "ไม่พบข้อมูลใน knowledge base ครับ" → STOP
    YES → D5
D5. Synthesize จาก chunks เท่านั้น ห้าม hallucinate จาก general knowledge
    ถ้าทั้ง 2 รอบพัง → "ไม่พบข้อมูลใน knowledge base ครับ"
D6. ตอบเสมอ ห้ามจบที่ tool result

## Silent failure — concrete example (ต้องทำแบบนี้เสมอ)

ตัวอย่าง fail→detect→correct:
  query: "สูตรต้มยำกุ้ง"
  rag_search return chunks เกี่ยวกับ "กองทุนรวม ASEAN" และ "การลงทุน"
  ❌ WRONG: synthesize คำตอบเรื่องอาหารจาก general knowledge (hallucination)
  ✅ CORRECT:
    D4 detect: chunks พูดถึงกองทุน ≠ query พูดถึงอาหาร → off-topic
    → ตอบ: "ไม่พบข้อมูลเรื่องการทำอาหารใน knowledge base ครับ knowledge base นี้มีข้อมูลเกี่ยวกับการลงทุนและการเงินเป็นหลัก"

ตัวอย่าง retry สำเร็จ:
  query: "value investing strategy"
  D1: rag_search("value investing strategy") → [low_quality]
  D3: rag_search("กลยุทธ์การลงทุนแบบเน้นคุณค่า") → chunks ตรงประเด็น
  D4: verify → ตรง → D5 synthesize → ตอบ

## RULE 6 — CALL save_memory เมื่อ user สั่งให้จำ:

  "จำไว้ว่า X"    → save_memory("X")
  "บันทึกไว้ว่า X" → save_memory("X")
  "จำไว้: X"      → save_memory("X")
  "remember X"    → save_memory("X")

CRITICAL: ส่ง X ทุกคำตามที่ user พูดไปใน save_memory — ห้ามแปล ห้าม paraphrase ห้ามตัดคำ
หลัง save_memory สำเร็จ ตอบสั้นๆ เช่น "บันทึกแล้วครับ" ไม่ต้องอธิบายเพิ่ม

## Multi-tool chaining — S-step: นับ tool verb ก่อนลงมือ

S1. SCAN — นับ tool verb ใน query:
      "หาไฟล์" / "ค้นหาไฟล์"  → search_files (1 call)
      "เปิดอ่าน" / "อ่านไฟล์" → read_file    (1 call)
      "หาข้อมูล" / "ค้นหา"    → rag_search  (1 call)
      "ดูรายการ" / "มีอะไรบ้าง" → list_knowledge (1 call)
S2. COUNT — จำนวน tool verb = จำนวน tool call ขั้นต่ำที่ต้องทำ
S3. EXECUTE — call ตามลำดับที่ปรากฏใน query ทีละขั้น
S4. RULE — context ≠ tool call. ถ้า user พูด verb ชัดเจน ต้อง call tool เสมอ แม้มีข้อมูลใน context แล้ว

ตัวอย่าง fail→detect→correct:

  query: "หาไฟล์ Behavioral แล้วอ่าน"
  S1 scan: "หาไฟล์" = search_files, "อ่าน" = read_file → 2 tools required
  ❌ WRONG: ตอบเนื้อหา Behavioral Finance จาก context — user ไม่ได้ขอสรุป ขอ tool calls
  ✅ CORRECT:
    1. search_files("Behavioral") → ได้รายการไฟล์ เช่น Behavioral_Finance.md
    2. read_file("Behavioral_Finance.md") → ได้ raw content
    3. ตอบจาก tool results

  query: "หาไฟล์ Strategy แล้วอ่านไฟล์แรก แล้วหาข้อมูล tactical allocation"
  S1 scan: "หาไฟล์" + "อ่านไฟล์" + "หาข้อมูล" → 3 tools required
  ❌ WRONG: rag_search("Strategy") ครั้งเดียวแล้วตอบ
  ✅ CORRECT:
    1. search_files("Strategy") → รายการไฟล์
    2. read_file(ไฟล์แรกในรายการ) → เนื้อหา
    3. rag_search("tactical asset allocation") → chunks
    4. synthesize จากผล 3 tool calls

  query: "ดูว่า knowledge base มีไฟล์อะไรบ้าง แล้วหาข้อมูลเรื่อง behavioral finance"
  S1 scan: "มีไฟล์อะไรบ้าง" = list_knowledge, "หาข้อมูล" = rag_search → 2 tools required
  ❌ WRONG: rag_search อย่างเดียว
  ✅ CORRECT:
    1. list_knowledge() → ภาพรวม KB
    2. rag_search("behavioral finance") → chunks
    3. ตอบรวม
"""

_log = logging.getLogger("chat")
_MEMORY_PATH = MEMORY_PATH


def _load_memory() -> str:
    if not _MEMORY_PATH.exists():
        return ""
    return _MEMORY_PATH.read_text(encoding="utf-8").strip()


def _build_prompt() -> str:
    memory = _load_memory()
    if not memory:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + f"\n\n## ความจำ (บันทึกจากครั้งก่อน)\n{memory}"


def _trim_history(history: list[dict]) -> list[dict]:
    pairs = len(history) // 2
    if pairs <= MAX_HISTORY_TURNS:
        return history
    return history[-(MAX_HISTORY_TURNS * 2):]


def _run_chat():
    ui.print_header(AGENT_MODEL)
    llm   = build_llm()
    agent = create_react_agent(llm, tools=[rag_search, list_knowledge, search_files, read_file, save_memory], prompt=_build_prompt())
    history: list[dict] = []

    while True:
        query = ui.prompt_user()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print(f"\n {ui.C_META}Goodbye{ui.R}\n")
            sys.exit(0)
        if query.lower() == "mode":
            ui.print_mode_menu()
            choice = ui.prompt_user()
            if choice == "2":
                _run_build()
                ui.print_header(AGENT_MODEL)
            elif choice in ("q", "quit", "exit"):
                print(f"\n {ui.C_META}Goodbye{ui.R}\n")
                sys.exit(0)
            continue

        # show user message
        print(f"\n {ui.C_USER}{ui.BOLD}You{ui.R}  {query}\n")

        sources_line = None
        scores: dict[str, float] = {}
        answer = ""
        t0 = time.time()

        with ui.Spinner("🧠 AI กำลังคิด…") as sp:
            # wire progress callback → spinner
            _progress.set_callback(lambda lbl, sub: (sp.update(lbl), sp.update_sub(sub)))

            trimmed  = _trim_history(history)
            messages = trimmed + [{"role": "user", "content": query}]
            try:
                from langchain_core.messages import AIMessage
                all_messages = []
                _cfg = {"recursion_limit": MAX_LOOP_ITERS * 2 + 1}
                for chunk in agent.stream({"messages": messages}, config=_cfg, stream_mode="updates"):
                    if "agent" in chunk:
                        msgs = chunk["agent"].get("messages", [])
                        all_messages.extend(msgs)
                        last = msgs[-1] if msgs else None
                        if last and getattr(last, "tool_calls", None):
                            sp.update("🔍 ค้นหาใน knowledge base…")
                            sp.update_sub("📝 สร้าง Q1+Q2+Q3…")
                        else:
                            sp.update("💬 สังเคราะห์คำตอบ…")
                            sp.update_sub("")
                    elif "tools" in chunk:
                        msgs = chunk["tools"].get("messages", [])
                        all_messages.extend(msgs)
                        sp.update("🧠 ประมวลผลผลลัพธ์…")
                        sp.update_sub("")

                for msg in reversed(all_messages):
                    content = str(getattr(msg, "content", ""))
                    if content and isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                        answer = content
                        break

                for msg in all_messages:
                    content = str(getattr(msg, "content", ""))
                    if "SOURCES:" in content:
                        for line in content.splitlines():
                            if line.startswith("SOURCES:"):
                                sources_line = line
                        scores = ui.extract_scores(content)

            except Exception as e:
                answer = f"เกิดข้อผิดพลาด: {e}"
            finally:
                _progress.set_callback(None)

        _log.info(f"[chat] done — {time.time()-t0:.1f}s")

        ui.print_answer(answer, sources_line, scores)
        ui.print_divider()

        history.append({"role": "user",      "content": query})
        history.append({"role": "assistant", "content": answer})
        history[:] = _trim_history(history)


def _run_build():
    log = logging.getLogger("build")
    files = [f for f in DATA_DIR.rglob("*") if _is_indexable_file(f)]

    ui.print_build_header(str(DATA_DIR), len(files))

    if not files:
        print(f"   {ui.C_WARN}No supported files found in knowledge/{ui.R}\n")
        return

    log.info(f"[build] ─── sync เริ่มต้น ───")
    progress    = ui.BuildProgress()
    build_start = time.time()

    # Purge registered files that no longer exist on disk
    current_abs = {str(f.resolve()) for f in files}
    for stale in all_registered():
        if stale not in current_abs:
            log.info(f"[build] purge deleted file: {Path(stale).name}")
            delete_file(Path(stale))
            deregister(stale)

    for idx, file in enumerate(sorted(files), 1):
        try:
            status = check(str(file))
            log.info(f"[build] [{idx}/{len(files)}] {file.name} → {status}")
            source = _ingestor_rel(file)

            if status == "skip":
                if store.has_source(source):
                    progress.add_row(file.name, "skip")
                    continue
                status = "changed"
                log.info(f"[build]   {file.name} — hash matched but data missing, re-ingesting")
            elif status == "new" and store.has_source(source):
                # Previous ingest died after Chroma upsert but before registry/
                # BM25 completion. Purge the partial source so semantic dedup
                # cannot hide the missing rows on retry.
                status = "changed"
                log.info(f"[build]   {file.name} — unregistered partial data found, rebuilding")

            if status == "changed":
                log.info(f"[build]   purge old chunks...")
                delete_file(file)
                deregister(str(file))

            with ui.Spinner(f"📄 {file.name}…"):
                n = ingest_file(file)
            register(str(file))
            progress.add_row(file.name, status, n)
        except Exception as e:
            log.error(f"[build]   {file.name} FAILED: {e}")
            progress.add_row(file.name, "error")

    elapsed = time.time() - build_start
    log.info(f"[build] ─── sync เสร็จ — {elapsed:.1f}s ───")
    progress.print_summary()

    issues = store.health_check()
    ghost_count = len(ghost_files())
    for issue in issues:
        log.warning(f"[build] anomaly: {issue}")
    if ghost_count:
        log.warning(f"[build] anomaly: {ghost_count} registered file(s) have zero chunks")
    ui.print_health_report(issues, ghost_count)


def main():
    ensure_runtime_dirs()
    with ui.Spinner("⚙️  กำลังโหลด AI model…") as sp:
        import embedder
        sp.update_sub("MiniLM embedder…")
        embedder._get_model()
        sp.update_sub("พร้อมแล้ว ✓")
        if not ensure_mlx_server(status_cb=sp.update_sub):
            print(f"\n {ui.C_WARN}⚠️  mlx_vlm.server ไม่ขึ้น — ดู {LOG_DIR / 'mlx_server.log'}{ui.R}\n")
    _run_chat()


if __name__ == "__main__":
    main()
