"""Upload an Excel/CSV file, profile it (rows, duplicates, missing values), and ask questions about it in plain English."""
import io, json, os, re
import numpy as np, pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine
from app import ai

router = APIRouter()
DB = "analysis.db"                       # separate from the reconciliation database
rw = create_engine(f"sqlite:///{DB}")
MAX_BYTES = 25 * 1024 * 1024
RESERVED = {"select", "from", "where", "group", "order", "by", "limit", "table", "index", "values", "key", "join", "on",
            "and", "or", "not", "null", "in", "is", "as", "case", "when", "then", "else", "end", "union", "all",
            "distinct", "having", "desc", "asc", "like", "between", "exists", "default", "check", "unique", "column",
            "add", "drop", "update", "insert", "delete", "create", "alter"}
BAD = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|truncate)\b", re.I)

SYSTEM = """You write SQLite queries for the user's uploaded data. Use ONLY these tables and exact column names:
{schema}
Rules: reply with ONE SELECT statement only (no markdown, no comments, no WITH/CTE; use subqueries instead).
Dates are TEXT like 'YYYY-MM-DD HH:MM:SS'. NULL means a missing value (use IS NULL, COUNT(col) vs COUNT(*)).
Duplicates: GROUP BY the relevant columns HAVING COUNT(*) > 1. Use AVG, MIN, MAX, SUM, COUNT as needed and ROUND(x, 2) for decimals.
For 'highest/lowest/top' use ORDER BY ... DESC/ASC with LIMIT."""

def ident(name, seen):
    s = re.sub(r"\W+", "_", str(name).strip().lower()).strip("_") or "col"
    if s[0].isdigit() or s in RESERVED:
        s = "c_" + s
    base, i = s, 2
    while s in seen:
        s, i = f"{base}_{i}", i + 1
    seen.add(s)
    return s

def _df(c, sql):
    r = c.exec_driver_sql(sql)
    return pd.DataFrame([tuple(x) for x in r.fetchall()], columns=list(r.keys()))

def _tables():
    if not os.path.exists(DB):
        return []
    with rw.connect() as c:
        return [r[0] for r in c.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table' AND name != '_meta'")]

def _read(name, content):
    low = name.lower()
    if low.endswith(".csv"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                return {"data": pd.read_csv(io.BytesIO(content), encoding=enc)}
            except UnicodeDecodeError:
                continue
    if low.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(content), sheet_name=None)
    raise HTTPException(400, "Please upload a .xlsx or .csv file (older .xls files: save as .xlsx first).")

def _js(o):
    return o.item() if hasattr(o, "item") else str(o)

def _profile(df):
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        info = {"name": c, "type": str(s.dtype), "missing": miss,
                "missing_pct": round(100 * miss / n, 1) if n else 0, "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().any():
            info.update(min=s.min(), max=s.max(), mean=round(float(s.mean()), 2), median=round(float(s.median()), 2))
        cols.append(info)
    dup = df.duplicated(keep="first")
    sample = df[df.duplicated(keep=False)].head(10)
    return json.loads(json.dumps({
        "rows": n, "columns": len(df.columns), "duplicate_rows": int(dup.sum()),
        "missing_cells": int(df.isna().sum().sum()), "column_info": cols,
        "duplicate_sample": json.loads(sample.to_json(orient="records", date_format="iso")),
    }, default=_js))

def _load_profile():
    ts = _tables()
    if not ts:
        raise HTTPException(400, "Upload a file first.")
    with rw.connect() as c:
        fname = _df(c, "SELECT filename FROM _meta")["filename"][0]
        out = [{"table": t, **_profile(_df(c, f'SELECT * FROM "{t}"'))} for t in ts]
    return {"filename": fname, "tables": out}

@router.get("/analyze")
def page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "analyze.html"))

@router.post("/files/upload")
def upload(file: UploadFile = File(...)):
    content = file.file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 25 MB.")
    try:
        sheets = _read(file.filename or "", content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read the file: {str(e)[:200]}")
    clean, seen_t = {}, set()
    for sheet, df in sheets.items():
        if df.empty:
            continue
        seen = set()
        df = df.copy()
        df.columns = [ident(c, seen) for c in df.columns]
        clean[ident(sheet, seen_t)] = df.replace(r"^\s*$", np.nan, regex=True)   # blank cells count as missing
    if not clean:
        raise HTTPException(400, "The file has no data.")
    old = _tables()
    with rw.begin() as c:
        for t in old:
            c.exec_driver_sql(f'DROP TABLE "{t}"')
        c.exec_driver_sql("DROP TABLE IF EXISTS _meta")
    for t, df in clean.items():
        df.to_sql(t, rw, index=False)
    pd.DataFrame([{"filename": file.filename}]).to_sql("_meta", rw, index=False)
    return _load_profile()

@router.get("/files/profile")
def profile():
    return _load_profile()

def _schema():
    parts = []
    with rw.connect() as c:
        for t in _tables():
            cols = c.exec_driver_sql(f'PRAGMA table_info("{t}")').fetchall()
            sample = json.loads(_df(c, f'SELECT * FROM "{t}" LIMIT 3').to_json(orient="records"))
            parts.append(f'Table "{t}": ' + ", ".join(f"{r[1]} ({r[2] or 'TEXT'})" for r in cols)
                         + f"\n  sample rows: {json.dumps(sample)}")
    return "\n".join(parts)

def check_sql(sql, allowed):
    s = sql.strip().rstrip(";").strip()
    if ";" in s: raise ValueError("Only one statement allowed.")
    if not re.match(r"(?is)^select\b", s): raise ValueError("Only SELECT is allowed.")
    if BAD.search(s) or "sqlite_" in s.lower(): raise ValueError("Query contains a forbidden keyword.")
    used = {t.strip('"').lower() for t in re.findall(r"(?:from|join)\s+([\w\"]+)", s, re.I)}
    if used - allowed: raise ValueError(f"Unknown table(s): {sorted(used - allowed)}")
    return s if re.search(r"\blimit\b", s, re.I) else s + " LIMIT 500"

def _run(sql):
    ro = create_engine(f"sqlite:///file:{DB}?mode=ro&uri=true")        # read-only at the database level
    try:
        with ro.connect() as c:
            r = c.exec_driver_sql(sql)
            return pd.DataFrame([tuple(x) for x in r.fetchmany(500)], columns=list(r.keys()))
    finally:
        ro.dispose()

def _explain(question, df):
    try:
        data = json.loads(df.head(20).to_json(orient="records"))
        return ai._ask("Answer the question in 1-2 plain sentences using ONLY the result rows given. "
                       "Do not invent numbers. If there are no rows, say no matching data was found.",
                       json.dumps({"question": question, "rows": data}), 1500).strip() or None
    except Exception:
        return None

class Q(BaseModel):
    question: str

@router.post("/files/ask")
def ask(q: Q):
    if not ai.has_key():
        raise HTTPException(400, "Set LLM_API_KEY to ask questions (the profile above works without it).")
    allowed = {t.lower() for t in _tables()}
    if not allowed:
        raise HTTPException(400, "Upload a file first.")
    system, prompt, sql, err, df = SYSTEM.format(schema=_schema()), q.question, "", "", None
    for _ in range(2):                                   # one automatic retry with the error message
        try:
            raw = ai._ask(system, prompt, 2000)
        except Exception as e:
            raise HTTPException(502, f"LLM call failed: {str(e)[:300]}")
        m = re.search(r"(?is)\bselect\b.*", raw)
        sql = re.sub(r"```.*$", "", m.group(0), flags=re.S).strip() if m else raw.strip()
        try:
            sql = check_sql(sql, allowed)
            df = _run(sql)
            break
        except Exception as e:
            err = str(e)
            prompt = f"{q.question}\n\nYour previous SQL failed.\nSQL: {sql}\nError: {err[:300]}\nReturn a corrected SELECT."
    if df is None:
        raise HTTPException(400, f"Query failed: {err[:200]} | SQL was: {sql}")
    return {"sql": sql, "columns": list(df.columns), "rows": json.loads(df.to_json(orient="records", date_format="iso")),
            "answer": _explain(q.question, df)}


# ---------------- Report dashboard (Power BI-style) ----------------
from fastapi.responses import Response

KEY_MEASURES = ["sale", "revenue", "amount", "profit", "cost", "quantity", "qty", "price", "total", "units", "income", "expense"]
KPI_KEYS = ["sale", "revenue", "amount", "profit", "cost", "quantity", "qty", "total", "units", "income", "expense"]
KEY_CATS = ["region", "category", "product", "segment", "customer", "country", "state", "city", "type", "department", "channel", "brand"]

def _rank(cols, keys):
    return sorted(cols, key=lambda c: min([i for i, k in enumerate(keys) if k in c] or [99]))

def _kinds(df):
    nums, cats, dates = [], [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if not re.search(r"(^|_)(id|code|no|number|zip|pincode|phone)$", c):
                nums.append(c)
            continue
        sample = s.dropna().astype(str).head(50)
        if sample.empty:
            continue
        iso = sample.str.match(r"^\d{4}-\d{2}-\d{2}").all()
        named = (not iso) and re.search(r"date|time|month|day", c) and pd.to_datetime(sample, errors="coerce", dayfirst=True).notna().mean() >= .9
        if iso or named:
            dates.append(c)
        elif 2 <= s.nunique() <= 50 and s.nunique() <= 0.5 * len(df):
            cats.append(c)
    return _rank(nums, KEY_MEASURES), _rank(cats, KEY_CATS), dates

def _group(df, by, measure, agg):
    if agg == "count" or not measure:
        return df.groupby(by).size()
    return getattr(df.groupby(by)[measure], "sum" if agg == "sum" else "mean")()

def _fmt(v):
    return f"{v:,.2f}".rstrip("0").rstrip(".")

@router.get("/files/report")
def report(table: str = "", measure: str = "", agg: str = "sum", category: str = "", date: str = "", share: str = ""):
    ts = _tables()
    if not ts:
        raise HTTPException(400, "Upload a file first.")
    t = table if table in ts else ts[0]
    with rw.connect() as c:
        df = _df(c, f'SELECT * FROM "{t}"')
    nums, cats, dates = _kinds(df)
    measure = measure if measure in nums else (nums[0] if nums else "")
    agg = agg if agg in ("sum", "avg", "count") and measure else "count" if not measure else "sum"
    category = category if category in cats else (cats[0] if cats else "")
    share = share if share in cats else (cats[1] if len(cats) > 1 else category)
    date = date if date in dates else (dates[0] if dates else "")
    out = {"options": {"tables": ts, "table": t, "measures": nums, "categories": cats, "dates": dates},
           "selected": {"measure": measure, "agg": agg, "category": category, "date": date, "share": share},
           "kpis": [{"label": "Rows", "value": len(df)}], "bar": None, "trend": None, "donut": None, "insights": []}
    # KPI cards: totals of the most business-like numeric columns (+ margin when both sales and profit exist)
    picks = [c for c in nums if any(k in c for k in KPI_KEYS)][:4] or nums[:3]
    for c in picks:
        out["kpis"].append({"label": f"Total {c}", "value": round(float(df[c].sum()), 2)})
    sales = next((c for c in nums if "sale" in c or "revenue" in c), None)
    profit = next((c for c in nums if "profit" in c), None)
    if sales and profit and df[sales].sum():
        out["kpis"].append({"label": "Profit margin %", "value": round(100 * float(df[profit].sum()) / float(df[sales].sum()), 1)})
    label = "Count of rows" if agg == "count" else f"{'Total' if agg == 'sum' else 'Average'} {measure}"
    out["label"] = label
    if category:
        g = _group(df.assign(_k=df[category].fillna("(missing)").astype(str)), "_k", measure, agg).sort_values(ascending=False).head(10)
        out["bar"] = {"title": f"{label} by {category}", "labels": list(g.index), "values": [round(float(v), 2) for v in g.values]}
        if agg != "avg" and g.sum():
            out["insights"].append(f"{g.index[0]} leads {category} with {_fmt(g.iloc[0])} ({100 * g.iloc[0] / _group(df.assign(_k=df[category].fillna('(missing)').astype(str)), '_k', measure, agg).sum():.0f}% of the total).")
    if date:
        d = pd.to_datetime(df[date], errors="coerce", dayfirst=not df[date].dropna().astype(str).str.match(r"^\d{4}-").all())
        g = _group(df.assign(_m=d.dt.to_period("M").astype(str)).loc[d.notna()], "_m", measure, agg).sort_index().tail(36)
        out["trend"] = {"title": f"{label} by month", "labels": list(g.index), "values": [round(float(v), 2) for v in g.values]}
        if len(g) >= 2:
            best = g.idxmax()
            out["insights"].append(f"Best month: {best} ({_fmt(g.max())}).")
            if g.iloc[-2]:
                out["insights"].append(f"Latest month {g.index[-1]} vs previous: {100 * (g.iloc[-1] - g.iloc[-2]) / abs(g.iloc[-2]):+.0f}%.")
    if share:
        sagg = "count" if agg == "count" else "sum"
        g = _group(df.assign(_k=df[share].fillna("(missing)").astype(str)), "_k", measure, sagg).sort_values(ascending=False)
        top = g.head(5)
        labels, values = list(top.index), [float(v) for v in top.values]
        if len(g) > 5:
            labels.append("Other"); values.append(float(g.iloc[5:].sum()))
        out["donut"] = {"title": f"Share of {'rows' if sagg == 'count' else measure} by {share}", "labels": labels, "values": [round(v, 2) for v in values]}
    dups = int(df.duplicated().sum())
    if dups:
        out["insights"].append(f"{dups} exact duplicate rows are counted in these totals; check the profile view before relying on them.")
    miss = int(df.isna().sum().sum())
    if miss:
        out["insights"].append(f"{miss} cells are missing values; totals ignore them.")
    return out

@router.get("/files/export")
def export(table: str = ""):
    ts = _tables()
    if not ts:
        raise HTTPException(400, "Upload a file first.")
    t = table if table in ts else ts[0]
    with rw.connect() as c:
        df = _df(c, f'SELECT * FROM "{t}"')
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{t}.csv"'})