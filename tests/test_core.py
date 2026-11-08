import pandas as pd, pytest
from app import matcher, ai

def df(rows):
    return pd.DataFrame(rows, columns=["reference", "merchant", "amount", "currency", "txn_date"])

def test_categories():
    L = df([("TXN1", "A", 100, "INR", "2026-09-01"), ("TXN2", "A", 50, "INR", "2026-09-01"),
            ("TXN3", "B", 80, "INR", "2026-09-01"), ("TXN4", "B", 70, "INR", "2026-09-01"),
            ("TXN5", "C", 10, "INR", "2026-09-01")])
    B = df([("txn-1", "A", 100, "INR", "2026-09-02"), ("TXN2", "A", 45, "INR", "2026-09-01"),
            ("TXN3", "B", 80, "INR", "2026-09-08"), ("TXN3", "B", 80, "INR", "2026-09-08"),
            ("TXN9", "D", 5, "INR", "2026-09-01")])
    r = matcher.reconcile(L, B).set_index(["key", "category"])
    cats = {k for k in r.index}
    assert ("TXN1", "matched") in cats            # reference formatting noise normalized
    assert ("TXN2", "amount_mismatch") in cats
    assert ("TXN3", "timing_delay") in cats and ("TXN3", "duplicate") in cats
    assert ("TXN4", "missing_in_bank") in cats and ("TXN9", "missing_in_ledger") in cats

def test_fuzzy_match():
    L = df([("TXN1", "A", 100, "INR", "2026-09-01")])
    B = df([("", "A", 100, "INR", "2026-09-02")])
    r = matcher.reconcile(L, B)
    assert r.category.tolist() == ["matched"] and bool(r.fuzzy.iloc[0])

@pytest.mark.parametrize("sql", ["DELETE FROM results", "SELECT 1; DROP TABLE results",
                                 "SELECT * FROM sqlite_master", "SELECT * FROM users", "PRAGMA table_info(results)"])
def test_sql_blocked(sql):
    with pytest.raises(ValueError):
        ai.validate_sql(sql)

def test_sql_limit_added():
    assert ai.validate_sql("SELECT * FROM results").endswith("LIMIT 100")
