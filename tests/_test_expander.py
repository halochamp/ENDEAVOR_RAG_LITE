# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""_test_expander.py — expander Thai detection, keyword extraction, contextualize, expand"""
from _runner import Runner

r = Runner("expander")


def t18_is_thai():
    from expander import _is_thai
    assert _is_thai("สวัสดี") is True
    assert _is_thai("hello") is False
    assert _is_thai("hello สวัสดี") is True


def t19_keywords_thai():
    from expander import _extract_keywords
    kw = _extract_keywords("การเรียนรู้ของเครื่องและปัญญาประดิษฐ์")
    assert isinstance(kw, str) and len(kw) > 0


def t20_keywords_english():
    from expander import _extract_keywords
    kw = _extract_keywords("what is machine learning and artificial intelligence")
    assert "machine" in kw or "learning" in kw


def t21_contextualize():
    from expander import maybe_contextualize
    history = [{"role": "user", "content": "ถามเรื่อง RAG"},
               {"role": "assistant", "content": "RAG คือ..."}]
    q = maybe_contextualize("แล้วมีประโยชน์ยังไง", history)
    assert "RAG" in q


def t22_translate():
    from expander import _translate
    result = _translate("RAG คืออะไร")
    assert isinstance(result, str)
    # may be empty if ollama unavailable — just no crash


def t23_expand_returns_3():
    from expander import expand
    q1, q2, q3 = expand("machine learning คืออะไร")
    assert isinstance(q1, str) and isinstance(q2, str) and isinstance(q3, str)
    assert q1 == "machine learning คืออะไร"


r.test("T18 _is_thai detection", t18_is_thai)
r.test("T19 Thai keyword extract", t19_keywords_thai)
r.test("T20 English keyword extract", t20_keywords_english)
r.test("T21 contextualize connector word", t21_contextualize)
r.test("T22 translate no crash", t22_translate)
r.test("T23 expand returns 3 strings", t23_expand_returns_3)

if __name__ == "__main__":
    r.exit()
