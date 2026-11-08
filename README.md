# Recon AI: payment reconciliation with AI exception triage

Matches an internal ledger against a bank/processor feed, flags mismatches, explains them with an LLM,
and answers plain-English questions through a validated, read-only text-to-SQL layer.

**Stack:** Python, FastAPI, pandas, scikit-learn (Isolation Forest), SQLAlchemy/SQLite (Postgres-ready), Claude API, pytest, Docker.

## Run it
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python generate_data.py                                # synthetic data + ground truth in data/
    uvicorn app.main:app --reload                          # open http://127.0.0.1:8000
    pytest -q                                              # tests
    python eval.py                                         # accuracy vs ground truth (after clicking Run reconciliation)
    # optional AI: copy .env.example to .env, set ANTHROPIC_API_KEY, then export it, and run: python eval.py --ai

Docker: `docker build -t recon-ai . && docker run -p 8000:8000 recon-ai`

## How it works
1. Pass 1 matches on a normalized reference; leftover rows get a fuzzy pass (merchant + amount + date window).
2. Classifies: matched, missing_in_bank, missing_in_ledger, amount_mismatch, duplicate, timing_delay.
3. Isolation Forest flags unusual amounts.
4. `/triage` asks the LLM for category, root cause, and action. The rule label is never sent, so accuracy is measurable.
5. `/ask` turns questions into SQL; `validate_sql` allows one SELECT on one table, adds a LIMIT, and runs on a read-only connection.

## Limitations (be upfront about these)
- All data is synthetic. Rule precision/recall is 100% partly because the generator and the rules share the same definitions; the LLM and text-to-SQL scores are the more informative numbers.
- SQLite by default; set DATABASE_URL for Postgres. No auth, no multi-user support.
