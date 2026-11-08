import json, os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from app import ai, matcher

engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///recon.db"))
app = FastAPI(title="Recon AI")
HERE = os.path.dirname(__file__)

def rows(df):
    return json.loads(df.to_json(orient="records"))

@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "static", "index.html"))

@app.post("/reconcile")
def reconcile():
    try:
        ledger, bank = pd.read_csv("data/ledger.csv"), pd.read_csv("data/bank.csv")
    except FileNotFoundError:
        raise HTTPException(400, "Run `python generate_data.py` first.")
    res = matcher.reconcile(ledger, bank)
    res.to_sql("results", engine, if_exists="replace", index=False)
    return {"ledger_rows": len(ledger), "bank_rows": len(bank), "result_rows": len(res)}

def _q(sql, **p):
    try:
        return pd.read_sql(text(sql), engine, params=p)
    except Exception:
        raise HTTPException(400, "No results yet. Click 'Run reconciliation' first.")

@app.get("/stats")
def stats():
    df = _q("SELECT category, COUNT(*) n, ROUND(SUM(ABS(diff)),2) value FROM results GROUP BY category")
    total = int(df["n"].sum())
    matched = int(df.loc[df.category == "matched", "n"].sum())
    anomalies = int(_q("SELECT COUNT(*) n FROM results WHERE anomaly=1")["n"][0])
    return {"total": total, "match_rate": round(100 * matched / total, 2), "anomalies": anomalies,
            "by_category": rows(df), "ai_enabled": ai.has_key()}

@app.get("/exceptions")
def exceptions(category: str = "", limit: int = 100):
    sql = "SELECT * FROM results WHERE category != 'matched'"
    if category:
        sql += " AND category = :c"
    return rows(_q(sql + " ORDER BY ABS(diff) DESC LIMIT :n", c=category, n=limit))

@app.post("/triage")
def triage(limit: int = 20):
    df = _q("SELECT * FROM results WHERE category != 'matched' AND ai_cause IS NULL ORDER BY ABS(diff) DESC LIMIT :n", n=limit)
    facts = [{"id": int(r.id), "in_ledger": bool(pd.notna(r.ledger_amount)),
              "in_bank": bool(pd.notna(r.bank_amount)), "ledger_amount": None if pd.isna(r.ledger_amount) else r.ledger_amount,
              "bank_amount": None if pd.isna(r.bank_amount) else r.bank_amount,
              "ledger_date": r.ledger_date, "bank_date": r.bank_date,
              "bank_rows_with_same_reference": 2 if r.category == "duplicate" else int(pd.notna(r.bank_amount)),
              "rule_category": r.category} for r in df.itertuples()]
    facts = json.loads(json.dumps(facts, default=lambda o: None if pd.isna(o) else float(o)))
    try:
        out = ai.triage(facts)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {str(e)[:300]}")
    with engine.begin() as c:
        for o in out:
            c.execute(text("UPDATE results SET ai_cause=:a, ai_action=:b WHERE id=:i"),
                      {"a": o["root_cause"], "b": o["action"], "i": o["id"]})
    return {"triaged": len(out), "mode": "llm" if ai.has_key() else "offline-rules"}

class Q(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Q):
    try:
        sql = ai.validate_sql(ai.text_to_sql(q.question))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {str(e)[:300]}")
    if not sql:
        raise HTTPException(500, "Validator returned no SQL. Your app/ai.py is out of date or was edited; re-copy it.")
    try:
        with engine.connect() as c:
            is_sqlite = engine.dialect.name == "sqlite"
            c.exec_driver_sql("PRAGMA query_only=ON" if is_sqlite else "SET TRANSACTION READ ONLY")  # read-only guard
            try:
                res = c.exec_driver_sql(sql)
                df = pd.DataFrame([tuple(r) for r in res.fetchall()], columns=list(res.keys()))
            finally:
                if is_sqlite:
                    c.exec_driver_sql("PRAGMA query_only=OFF")   # connections are pooled: always reset
    except Exception as e:
        raise HTTPException(400, f"Query failed: {str(e)[:200]} | SQL was: {sql}")
    return {"sql": sql, "rows": rows(df)}