import re
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

AMT_TOL, DATE_TOL = 0.005, 2

def norm(ref):
    return re.sub(r"[^A-Z0-9]", "", str(ref).upper()) if pd.notna(ref) else ""

def reconcile(ledger: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
    L, B = ledger.copy(), bank.copy()
    for df in (L, B):
        df["key"] = df["reference"].map(norm)
        df["txn_date"] = pd.to_datetime(df["txn_date"])
    dup_mask = B.duplicated("key", keep="first") & (B["key"] != "")
    D, B1 = B[dup_mask], B[~dup_mask]

    # Pass 1: exact match on normalized reference
    m = L.merge(B1, on="key", how="outer", suffixes=("_l", "_b"), indicator=True)
    m["_merge"] = m["_merge"].astype(str)
    m["fuzzy"] = False

    # Pass 2: fuzzy match leftovers on merchant + amount + date window
    lo = m[m._merge == "left_only"].reset_index()
    ro = m[m._merge == "right_only"].reset_index()
    p = lo.merge(ro, left_on=["merchant_l", "amount_l"], right_on=["merchant_b", "amount_b"], suffixes=("_lo", "_ro"))
    p = p[(p["txn_date_l_lo"] - p["txn_date_b_ro"]).abs() <= pd.Timedelta(days=DATE_TOL)]
    p = p.drop_duplicates("index_lo").drop_duplicates("index_ro")
    for li, ri in zip(p["index_lo"], p["index_ro"]):
        for c in ("reference", "merchant", "amount", "txn_date", "currency"):
            m.loc[li, c + "_b"] = m.loc[ri, c + "_b"]
        m.loc[li, "_merge"] = "both"
        m.loc[li, "fuzzy"] = True
    m = m.drop(index=list(p["index_ro"]))

    def classify(r):
        if r["_merge"] == "left_only": return "missing_in_bank"
        if r["_merge"] == "right_only": return "missing_in_ledger"
        if abs(r["amount_l"] - r["amount_b"]) > AMT_TOL: return "amount_mismatch"
        if abs((r["txn_date_b"] - r["txn_date_l"]).days) > DATE_TOL: return "timing_delay"
        return "matched"

    out = pd.DataFrame({
        "key": m["key"], "category": m.apply(classify, axis=1),
        "merchant": m["merchant_l"].fillna(m["merchant_b"]),
        "ledger_amount": m["amount_l"], "bank_amount": m["amount_b"],
        "ledger_date": m["txn_date_l"], "bank_date": m["txn_date_b"],
        "fuzzy": m["fuzzy"],
    })
    lamt = L.drop_duplicates("key").set_index("key")["amount"]
    dups = pd.DataFrame({
        "key": D["key"], "category": "duplicate", "merchant": D["merchant"],
        "ledger_amount": D["key"].map(lamt), "bank_amount": D["amount"],
        "ledger_date": pd.NaT, "bank_date": D["txn_date"], "fuzzy": False,
    })
    out = pd.concat([out, dups], ignore_index=True)
    out["diff"] = (out["bank_amount"].fillna(0) - out["ledger_amount"].fillna(0)).round(2)
    out["anomaly"] = flag_anomalies(out)
    for c in ("ledger_date", "bank_date"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    out["ai_cause"], out["ai_action"] = None, None
    out.insert(0, "id", range(len(out)))
    return out

def flag_anomalies(df: pd.DataFrame) -> np.ndarray:
    """Unsupervised outlier detection on transaction amount (Isolation Forest)."""
    x = np.log1p(df[["ledger_amount", "bank_amount"]].max(axis=1).fillna(0).to_numpy(dtype=float)).reshape(-1, 1)
    return (IsolationForest(contamination=0.005, random_state=42).fit_predict(x) == -1).astype(int)
