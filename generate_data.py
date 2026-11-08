"""Generate synthetic ledger + bank feeds with injected, labeled errors."""
import os, random
from datetime import date, timedelta
import pandas as pd
from faker import Faker

def build(n=5000, out="data", seed=42):
    random.seed(seed); Faker.seed(seed); fake = Faker()
    os.makedirs(out, exist_ok=True)
    merchants = [fake.company() for _ in range(25)]
    start = date(2026, 9, 1)
    ledger, bank, truth = [], [], []
    for i in range(n):
        ref = f"TXN{100000 + i}"
        m = random.choice(merchants)
        amt = max(20.0, round(random.lognormvariate(5, 1), 2))
        d = start + timedelta(days=random.randint(0, 59))
        L = dict(reference=ref, merchant=m, amount=amt, currency="INR", txn_date=d.isoformat())
        B = dict(L, txn_date=(d + timedelta(days=random.randint(0, 1))).isoformat())
        cat, r = "matched", random.random()
        extra = None
        if r < .03: cat = "missing_in_bank"; B = None
        elif r < .05: cat = "missing_in_ledger"; L = None
        elif r < .08:
            cat = "amount_mismatch"; B["amount"] = round(amt * random.choice([.98, .95, 1.05]), 2)
        elif r < .10: cat = "duplicate"; extra = dict(B)
        elif r < .12:
            cat = "timing_delay"; B["txn_date"] = (d + timedelta(days=random.randint(4, 7))).isoformat()
        elif r < .123:
            cat = "outlier"; big = round(amt * random.randint(100, 200), 2); L["amount"] = B["amount"] = big
        elif r < .133: cat = "matched"; B["reference"] = ""          # reference lost -> fuzzy match
        elif r < .163: B["reference"] = ref.lower().replace("txn", "txn-")  # formatting noise
        if L: ledger.append(L)
        if B: bank.append(B)
        if extra: bank.append(extra)
        truth.append(dict(reference=ref, true_category=cat))
    pd.DataFrame(ledger).to_csv(f"{out}/ledger.csv", index=False)
    pd.DataFrame(bank).sample(frac=1, random_state=seed).to_csv(f"{out}/bank.csv", index=False)
    pd.DataFrame(truth).to_csv(f"{out}/ground_truth.csv", index=False)
    print(f"ledger={len(ledger)} bank={len(bank)} -> {out}/")

if __name__ == "__main__":
    build()
