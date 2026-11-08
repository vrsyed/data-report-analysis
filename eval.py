"""Score the system against ground truth. `python eval.py` (rules) | `python eval.py --ai` (adds LLM checks)."""
import sys, json
import pandas as pd
from sqlalchemy import create_engine, text
from app import ai

engine = create_engine("sqlite:///recon.db")
res = pd.read_sql("SELECT * FROM results", engine)
truth = pd.read_csv("data/ground_truth.csv")
cats = [c for c in ai.CATS if c != "matched"]

pred = set(map(tuple, res[res.category != "matched"][["key", "category"]].values))
real = set(map(tuple, truth[truth.true_category.isin(cats)][["reference", "true_category"]].values))
print("\n== Rule-based exception detection ==")
rows = []
for c in cats:
    p, r = {x for x in pred if x[1] == c}, {x for x in real if x[1] == c}
    tp = len(p & r)
    rows.append((c, len(r), round(tp / len(p), 3) if p else 0, round(tp / len(r), 3) if r else 0))
print(pd.DataFrame(rows, columns=["category", "true_count", "precision", "recall"]).to_string(index=False))
outl = set(truth[truth.true_category == "outlier"].reference)
flagged = set(res[res.anomaly == 1].key)
print(f"\nIsolation Forest: caught {len(outl & flagged)}/{len(outl)} injected outliers, {len(flagged - outl)} other rows flagged")
fz = int(res.fuzzy.sum()); print(f"Fuzzy-matched pairs: {fz}")

if "--ai" in sys.argv:
    print("\n== LLM triage accuracy (blind: model never sees the rule label) ==")
    ex = res[res.category != "matched"].sample(40, random_state=1)
    tmap = dict(zip(truth.reference, truth.true_category))
    facts = [dict(id=int(r.id), in_ledger=bool(pd.notna(r.ledger_amount)), in_bank=bool(pd.notna(r.bank_amount)),
                  ledger_amount=None if pd.isna(r.ledger_amount) else r.ledger_amount,
                  bank_amount=None if pd.isna(r.bank_amount) else r.bank_amount,
                  ledger_date=r.ledger_date, bank_date=r.bank_date,
                  bank_rows_with_same_reference=2 if r.category == "duplicate" else int(pd.notna(r.bank_amount)),
                  rule_category=r.category) for r in ex.itertuples()]
    facts = json.loads(json.dumps(facts, default=lambda o: None if pd.isna(o) else float(o)))
    out = {o["id"]: o["category_guess"] for o in ai.triage(facts)}
    ok = sum(out.get(int(r.id)) == tmap[r.key] for r in ex.itertuples())
    print(f"Accuracy: {ok}/{len(ex)} = {ok / len(ex):.0%}")

    print("\n== Natural-language-to-SQL (result must equal the golden query) ==")
    golden = {
        "How many duplicate rows are there?": "SELECT COUNT(*) FROM results WHERE category='duplicate'",
        "What is the total absolute difference for amount mismatches?": "SELECT ROUND(SUM(ABS(diff)),2) FROM results WHERE category='amount_mismatch'",
        "Which merchant has the most missing_in_bank rows?": "SELECT merchant FROM results WHERE category='missing_in_bank' GROUP BY merchant ORDER BY COUNT(*) DESC LIMIT 1",
        "How many rows are flagged as anomalies?": "SELECT COUNT(*) FROM results WHERE anomaly=1",
    }
    good = 0
    for q, g in golden.items():
        try:
            got = pd.read_sql(text(ai.validate_sql(ai.text_to_sql(q))), engine).values.tolist()
        except Exception as e:
            got = str(e)
        exp = pd.read_sql(text(g), engine).values.tolist()
        good += got == exp
        print(("PASS " if got == exp else "FAIL ") + q)
    print(f"SQL accuracy: {good}/{len(golden)}")
