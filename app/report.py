"""Power BI-style report engine: slicers (filters), DAX-style measures, RANKX, and recommendations."""
import ast, json, operator
from types import SimpleNamespace
import numpy as np, pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app import ai
from app.files import _df, _fmt, _kinds, _tables, rw

router = APIRouter()
AGGS = {"sum": "SUM", "avg": "AVERAGE", "min": "MIN", "max": "MAX", "median": "MEDIAN", "count": "COUNTROWS",
        "distinct": "DISTINCTCOUNT", "sumx": "SUMX", "avgx": "AVERAGEX", "divide": "DIVIDE"}
NUMERIC = {"sum", "avg", "min", "max", "median"}
ADDITIVE = {"sum", "sumx", "count"}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

class Req(BaseModel):
    table: str = ""
    agg: str = "sum"
    measure: str = ""
    expr: str = ""
    category: str = ""
    date: str = ""
    share: str = ""
    top: int = 10
    sort: str = "desc"
    trend_mode: str = "value"
    filters: dict = {}
    ranges: dict = {}

def _dates(s):
    iso = s.dropna().astype(str).str.match(r"^\d{4}-").all()
    return pd.to_datetime(s, errors="coerce", dayfirst=not iso)

def _num(v):
    return None if v is None or pd.isna(v) or np.isinf(v) else round(float(v), 2)

def _eval(expr, df, nums):
    """Safe formula evaluator: column names, numbers, + - * / and parentheses only."""
    used = set()
    def ev(n):
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in nums:
                raise ValueError(f"'{n.id}' is not a numeric column. Numeric columns: {', '.join(nums)}")
            used.add(n.id)
            return df[n.id]
        raise ValueError("Formulas can use only column names, numbers, + - * / and parentheses.")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        raise ValueError("Could not read the formula. Example: quantity * unit_price")
    res = ev(tree.body)
    if not used:
        raise ValueError("The formula must use at least one column.")
    return pd.Series(res, index=df.index).replace([np.inf, -np.inf], np.nan)

def _spec(req, nums, allcols):
    agg = req.agg if req.agg in AGGS else "sum"
    m = req.measure
    if agg in NUMERIC:
        m = m if m in nums else (nums[0] if nums else "")
        if not m:
            agg = "count"
    elif agg == "distinct":
        m = m if m in allcols else allcols[0]
    else:
        m = ""
    return SimpleNamespace(agg=agg, measure=m, expr=req.expr.strip())

def _label(s):
    if s.agg == "count": return "COUNTROWS"
    if s.agg in ("sumx", "avgx"): return f"{AGGS[s.agg]}({s.expr})"
    if s.agg == "divide":
        parts = [p.strip() for p in s.expr.split(";")]
        return f"DIVIDE(SUM({parts[0]}), SUM({parts[1]}))" if len(parts) == 2 else "DIVIDE"
    return f"{AGGS[s.agg]}({s.measure})"

def _frame(f, s, nums):
    if s.agg in ("sumx", "avgx"):
        return f.assign(_v=_eval(s.expr, f, nums))
    if s.agg == "divide":
        parts = s.expr.split(";")
        if len(parts) != 2:
            raise ValueError("DIVIDE needs two formulas separated by ';', for example: profit ; sales")
        return f.assign(_v=_eval(parts[0], f, nums), _b=_eval(parts[1], f, nums))
    return f

def _agg(frame, by, s):
    g = frame.groupby(by)
    if s.agg == "count": return g.size()
    if s.agg == "distinct": return g[s.measure].nunique()
    if s.agg == "divide":
        t = g[["_v", "_b"]].sum()
        return t["_v"] / t["_b"].where(t["_b"] != 0)
    col = "_v" if s.agg in ("sumx", "avgx") else s.measure
    fn = {"sum": "sum", "sumx": "sum", "avg": "mean", "avgx": "mean", "min": "min", "max": "max", "median": "median"}[s.agg]
    return getattr(g[col], fn)()

def _total(frame, s):
    r = _agg(frame, np.zeros(len(frame), dtype=int), s)
    return _num(r.iloc[0]) if len(r) else 0

def _filter(df, req, dates):
    mask = pd.Series(True, index=df.index)
    for col, vals in (req.filters or {}).items():
        if col in df.columns and vals:
            mask &= df[col].fillna("(missing)").astype(str).isin([str(v) for v in vals])
    for col, rng in (req.ranges or {}).items():
        if col in dates and isinstance(rng, list) and len(rng) == 2:
            try:
                d = _dates(df[col])
                if rng[0]: mask &= d >= pd.to_datetime(rng[0])
                if rng[1]: mask &= d < pd.to_datetime(rng[1]) + pd.Timedelta(days=1)
            except (ValueError, TypeError):
                pass
    return df[mask]

def _load(req):
    ts = _tables()
    if not ts:
        raise HTTPException(400, "Upload a file first.")
    t = req.table if req.table in ts else ts[0]
    with rw.connect() as c:
        return t, ts, _df(c, f'SELECT * FROM "{t}"')

def _recs(f, nums, cats, dates):
    """Rule-based recommendations from patterns in the (filtered) data. Returns (list, facts for the AI)."""
    rev = (next((c for c in nums if any(k in c for k in ("sale", "revenue", "amount"))), None)
           or next((c for c in nums if "total" in c), None) or (nums[0] if nums else None))
    prof = next((c for c in nums if "profit" in c), None)
    if rev is None or len(f) < 5 or not f[rev].sum():
        return [], {}
    total = float(f[rev].sum())
    om = float(f[prof].sum()) / total * 100 if prof else None
    recs, facts = [], {"revenue_column": rev, "profit_column": prof, "rows": len(f), "total_revenue": round(total, 2),
                       "overall_margin_pct": None if om is None else round(om, 1), "dimensions": {}}
    for dim in cats[:3]:
        key = f[dim].fillna("(missing)").astype(str)
        t = f.assign(_k=key)
        g = t.groupby("_k").agg(revenue=(rev, "sum"), rows=(rev, "size"))
        if prof: g["profit"] = t.groupby("_k")[prof].sum()
        g["share"] = g.revenue / total * 100
        if prof: g["margin"] = g.profit / g.revenue.where(g.revenue != 0) * 100
        if "(missing)" in g.index and g.loc["(missing)", "rows"] / len(f) > 0.03:
            miss = g.loc["(missing)"]
            recs.append({"type": "data", "text": f"{dim} is blank in {100 * miss.rows / len(f):.0f}% of rows ({_fmt(miss.revenue)} of {rev} cannot be attributed). Fix data capture so {dim}-level decisions are reliable."})
        gm = g.drop("(missing)", errors="ignore")
        if len(gm) < 2: continue
        avg_share = 100 / len(gm)
        top, bot = gm.revenue.idxmax(), gm.revenue.idxmin()
        facts["dimensions"][dim] = [{"name": k, "revenue": round(float(r.revenue), 2), "share_pct": round(float(r.share), 1),
                                     **({"margin_pct": round(float(r.margin), 1)} if prof and pd.notna(r.margin) else {})}
                                    for k, r in gm.sort_values("revenue", ascending=False).iterrows()][:8]
        recs.append({"type": "grow", "text": f"{top} is the top {dim} with {_fmt(gm.loc[top, 'revenue'])} {rev} ({gm.loc[top, 'share']:.0f}% of total). Protect it: keep stock and service levels high, and use it as the anchor for bundles and cross-sell."})
        if gm.loc[bot, "share"] < 0.75 * avg_share:
            recs.append({"type": "fix", "text": f"{bot} is clearly the weakest {dim} ({_fmt(gm.loc[bot, 'revenue'])}, {gm.loc[bot, 'share']:.0f}% vs an even split of {avg_share:.0f}%). Find out why (pricing, availability, visibility), then run a focused promotion or reduce effort here."})
        elif gm.loc[top, "share"] < 1.25 * avg_share:
            recs.append({"type": "info", "text": f"{dim} is fairly evenly spread (top {gm.loc[top, 'share']:.0f}% vs lowest {gm.loc[bot, 'share']:.0f}%), so there is no single weak spot. Look at margin and trend for improvement instead."})
        if prof and gm.margin.notna().any():
            hi, lo = gm.margin.idxmax(), gm.margin.idxmin()
            if om is not None and gm.loc[hi, "margin"] >= om + 2 and gm.loc[hi, "share"] < avg_share:
                recs.append({"type": "grow", "text": f"{hi} has the best margin ({gm.loc[hi, 'margin']:.0f}% vs {om:.0f}% overall) but only {gm.loc[hi, 'share']:.0f}% of sales. Grow it: more promotion and placement, and offer it as the upgrade or add-on."})
            if om is not None and gm.loc[lo, "share"] >= avg_share and gm.loc[lo, "margin"] < om - 2:
                recs.append({"type": "fix", "text": f"{lo} brings {gm.loc[lo, 'share']:.0f}% of sales but a thin {gm.loc[lo, 'margin']:.0f}% margin (overall {om:.0f}%). Review discounts and costs before pushing more volume."})
        s2 = gm.share.sort_values(ascending=False).head(2).sum()
        if len(gm) >= 5 and s2 > 50:
            recs.append({"type": "risk", "text": f"Two {dim} values ({', '.join(gm.share.sort_values(ascending=False).head(2).index)}) account for {s2:.0f}% of {rev}. That is a concentration risk; build up the others."})
    for dc in dates[:1]:
        d = _dates(f[dc])
        m = f.assign(_m=d.dt.to_period("M").astype(str)).loc[d.notna()].groupby("_m")[rev].sum().sort_index()
        if len(m) >= 3:
            facts["monthly_revenue"] = {k: round(float(v), 2) for k, v in m.tail(12).items()}
            recs.append({"type": "trend", "text": f"Seasonality: best month {m.idxmax()} ({_fmt(m.max())}), weakest {m.idxmin()} ({_fmt(m.min())}). Plan stock and campaigns ahead of the peak and run offers in the weak months."})
        if len(m) >= 6:
            last, prev = m.iloc[-3:].sum(), m.iloc[-6:-3].sum()
            if prev:
                ch = 100 * (last - prev) / prev
                recs.append({"type": "trend", "text": f"The last 3 months are {ch:+.0f}% vs the 3 months before. " + ("Keep the momentum: find what changed and repeat it." if ch >= 0 else "Investigate what changed (pricing, stock-outs, lost customers).")})
            lastd = d.max()
            if pd.notna(lastd) and lastd.day < 25:
                recs.append({"type": "data", "text": f"The latest month ({m.index[-1]}) may be incomplete (data ends on day {lastd.day}), so its numbers can look lower than reality."})
    dups = int(f.duplicated().sum())
    if dups:
        recs.append({"type": "data", "text": f"{dups} exact duplicate rows are counted in these numbers. Remove them before acting on the totals."})
    order = {"fix": 0, "grow": 1, "risk": 2, "trend": 3, "data": 4, "info": 5}
    recs.sort(key=lambda r: order[r["type"]])
    return recs[:10], facts

@router.post("/api/report")
def report(req: Req):
    t, ts, df = _load(req)
    nums, cats, dates = _kinds(df)
    allcols = list(df.columns)
    s = _spec(req, nums, allcols)
    category = req.category if req.category in cats else (cats[0] if cats else "")
    share = req.share if req.share in cats else (cats[1] if len(cats) > 1 else category)
    date = req.date if req.date in dates else (dates[0] if dates else "")
    top = min(max(int(req.top), 3), 30)
    sort = "asc" if req.sort == "asc" else "desc"
    tmode = req.trend_mode if req.trend_mode in ("value", "cumulative", "mom") else "value"
    out = {"options": {"tables": ts, "table": t, "measures": nums, "columns": allcols, "categories": cats, "dates": dates,
                       "aggs": [[k, v] for k, v in AGGS.items()]},
           "selected": {"table": t, "agg": s.agg, "measure": s.measure, "expr": s.expr, "category": category, "date": date,
                        "share": share, "top": top, "sort": sort, "trend_mode": tmode},
           "slicers": {c: list(df[c].fillna("(missing)").astype(str).value_counts().index[:30]) for c in cats},
           "date_ranges": {c: [str(_dates(df[c]).min())[:10], str(_dates(df[c]).max())[:10]] for c in dates},
           "kpis": [], "bar": None, "trend": None, "donut": None, "ranking": [], "recs": [], "data": None,
           "rows": {"shown": 0, "total": len(df)}, "label": "", "error": None}
    f = _filter(df, req, dates)
    out["rows"]["shown"] = len(f)
    if f.empty:
        out["error"] = "No rows match the current slicers. Clear a filter to see data again."
        return out
    try:
        frame = _frame(f, s, nums)
    except ValueError as e:
        out["error"] = str(e)
        return out
    label = _label(s)
    out["label"] = label
    out["kpis"].append({"label": "Rows", "value": len(f), "sub": f"of {len(df)}" if len(f) != len(df) else ""})
    out["kpis"].append({"label": label, "value": _total(frame, s), "sub": "selected measure"})
    picks = [c for c in nums if any(k in c for k in ("sale", "revenue", "amount", "profit", "cost", "quantity", "qty", "units"))][:3]
    for c in picks:
        out["kpis"].append({"label": f"Total {c}", "value": _num(f[c].sum()), "sub": ""})
    sc = next((c for c in nums if "sale" in c or "revenue" in c), None)
    pc = next((c for c in nums if "profit" in c), None)
    if sc and pc and f[sc].sum():
        out["kpis"].append({"label": "Profit margin %", "value": round(100 * float(f[pc].sum()) / float(f[sc].sum()), 1), "sub": ""})
    if category:
        key = frame[category].fillna("(missing)").astype(str)
        g = _agg(frame.assign(_k=key), "_k", s).dropna()
        gs = g.sort_values(ascending=(sort == "asc")).head(top)
        out["bar"] = {"title": f"{label} by {category}", "column": category, "labels": list(gs.index), "values": [_num(v) for v in gs.values]}
        rk = g.rank(method="min", ascending=False)
        tot = g.sum() if s.agg in ADDITIVE else None
        cum = g.sort_values(ascending=False).cumsum() if tot else None
        for k in gs.index:
            out["ranking"].append({"rank": int(rk[k]), "name": k, "value": _num(g[k]),
                                   "share": _num(100 * g[k] / tot) if tot else None,
                                   "cumulative": _num(100 * cum[k] / tot) if tot else None})
    if date:
        d = _dates(frame[date])
        tmp = frame.assign(_m=d.dt.to_period("M").astype(str)).loc[d.notna()]
        g = _agg(tmp, "_m", s).sort_index().tail(36) if len(tmp) else pd.Series(dtype=float)
        title = f"{label} by month"
        if tmode == "cumulative" and s.agg in ADDITIVE:
            g, title = g.cumsum(), f"Cumulative {label} by month"
        elif tmode == "mom":
            g, title = (g.pct_change() * 100).replace([np.inf, -np.inf], np.nan).dropna(), f"{label}: month-over-month change %"
        out["trend"] = {"title": title, "labels": list(g.index), "values": [_num(v) for v in g.values]}
    if share:
        ds = s if s.agg in ADDITIVE else (SimpleNamespace(agg="sum", measure=s.measure, expr="") if s.agg in NUMERIC else SimpleNamespace(agg="count", measure="", expr=""))
        dframe = frame if ds is s else f
        g = _agg(dframe.assign(_k=f[share].fillna("(missing)").astype(str)), "_k", ds).sort_values(ascending=False)
        topg = g.head(5)
        labels, vals = list(topg.index), [float(v) for v in topg.values]
        if len(g) > 5:
            labels.append("Other"); vals.append(float(g.iloc[5:].sum()))
        out["donut"] = {"title": f"Share of {'rows' if ds.agg == 'count' else _label(ds)} by {share}", "labels": labels, "values": [round(v, 2) for v in vals]}
    out["recs"] = _recs(f, nums, cats, dates)[0]
    out["data"] = {"columns": list(f.columns), "rows": json.loads(f.head(50).to_json(orient="records", date_format="iso"))}
    return out

RECO_SYSTEM = ("You are a sales and profitability analyst. Using ONLY the JSON facts provided, write 5 concise numbered "
               "recommendations to increase sales and profit. Each must name specific products/regions/customers/months from the "
               "facts, cite a number, say what to do and why. Say so if evidence is thin. Plain text, no tables, no invented data.")

@router.post("/api/recommend")
def recommend(req: Req):
    if not ai.has_key():
        raise HTTPException(400, "Set LLM_API_KEY to generate AI recommendations. The rule-based suggestions already work without it.")
    t, ts, df = _load(req)
    nums, cats, dates = _kinds(df)
    recs, facts = _recs(_filter(df, req, dates), nums, cats, dates)
    if not facts:
        raise HTTPException(400, "Not enough data: need at least 5 rows and a numeric sales column.")
    facts["rule_based_findings"] = [r["text"] for r in recs]
    try:
        return {"text": ai._ask(RECO_SYSTEM, json.dumps(facts), 2500).strip()}
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {str(e)[:300]}")