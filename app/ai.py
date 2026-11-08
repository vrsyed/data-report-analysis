import json, os, re

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CATS = ["matched", "missing_in_bank", "missing_in_ledger", "amount_mismatch", "duplicate", "timing_delay"]
FALLBACK = {
    "missing_in_bank": ("Ledger entry has no bank record; settlement may be pending or failed.", "Check processor settlement report; contact bank if older than 3 days."),
    "missing_in_ledger": ("Bank received funds with no ledger entry; possible unrecorded sale or misdirected payment.", "Search ledger by merchant/amount; book entry or raise a query with the bank."),
    "amount_mismatch": ("Bank amount differs from ledger; likely fees, FX, or keying error.", "Compare against fee schedule; correct the ledger or dispute with the bank."),
    "duplicate": ("Same reference appears twice in the bank feed; likely duplicate posting.", "Request reversal of the duplicate from the bank."),
    "timing_delay": ("Dates differ by more than the tolerance window; settlement lag.", "Confirm settlement date; no correction needed if amounts agree."),
}
SCHEMA = ("Table results(id INTEGER, key TEXT, category TEXT, merchant TEXT, ledger_amount REAL, bank_amount REAL, "
          "ledger_date TEXT, bank_date TEXT, diff REAL, anomaly INTEGER, ai_cause TEXT, ai_action TEXT). "
          f"category is one of {CATS}. Dates are YYYY-MM-DD text.")

def _cfg():
    """Free path: any OpenAI-compatible provider (Groq default). Optional paid path: Anthropic."""
    if os.getenv("LLM_API_KEY"):
        return ("openai", os.environ["LLM_API_KEY"],
                os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
    if os.getenv("ANTHROPIC_API_KEY"):
        return ("anthropic", os.environ["ANTHROPIC_API_KEY"], None, MODEL)
    return None

def has_key():
    return _cfg() is not None

def _ask(system, user, max_tokens=6000):
    kind, key, base, model = _cfg()
    if kind == "anthropic":
        import anthropic
        r = anthropic.Anthropic(api_key=key).messages.create(model=model, max_tokens=max_tokens, system=system,
                                                             messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")
    from openai import OpenAI
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}   # reasoning models: keep thinking short
    r = OpenAI(api_key=key, base_url=base).chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0, **extra,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content or ""

def _json(text):
    a, b = text.find("["), text.rfind("]")
    return json.loads(text[a:b + 1])

def triage(rows):
    """rows: dicts with id + raw facts + rule_category (rule_category is never sent to the model)."""
    if not has_key():
        return [{"id": r["id"], "category_guess": r["rule_category"], "root_cause": FALLBACK[r["rule_category"]][0],
                 "action": FALLBACK[r["rule_category"]][1]} for r in rows]
    facts = [{k: v for k, v in r.items() if k != "rule_category"} for r in rows]
    system = ("You are a payments reconciliation analyst. For each item decide the exception category "
              f"from {CATS[1:]} using only the facts given (amounts must match exactly; dates within 2 days are fine). "
              "Reply with ONLY a JSON list of objects: id, category_guess, root_cause (1 sentence), action (1 sentence).")
    return _json(_ask(system, json.dumps(facts)))

def text_to_sql(question):
    if not has_key():
        raise RuntimeError("Set LLM_API_KEY (free Groq key) to use natural-language questions.")
    system = ("Translate the question into ONE read-only SQLite SELECT statement over this schema. "
              f"{SCHEMA} Use only the table 'results'. Reply with the SQL only, no explanation, no markdown.")
    out = _ask(system, question, 2000).strip()
    m = re.search(r"(?is)\bselect\b.*", out)
    return re.sub(r"```.*$", "", m.group(0), flags=re.S).strip() if m else out

BAD = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|truncate|grant)\b", re.I)

def validate_sql(sql):
    s = sql.strip().rstrip(";").strip()
    if ";" in s: raise ValueError("Only one statement allowed.")
    if not re.match(r"(?is)^select\b", s): raise ValueError("Only SELECT is allowed.")
    if BAD.search(s) or "sqlite_" in s.lower(): raise ValueError("Query contains a forbidden keyword.")
    tables = {t.strip('"').lower() for t in re.findall(r"(?:from|join)\s+([\w\"]+)", s, re.I)}
    if tables - {"results"}: raise ValueError(f"Unknown table(s): {tables - {'results'}}")
    has_limit = re.search(r"\blimit\b", s, re.I) is not None
    return s if has_limit else s + " LIMIT 100"  