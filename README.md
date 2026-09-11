# Spaid Analytics

A private stock-recommendation and portfolio-decision engine for the S&P 500.

It answers one question first: **what should I buy today?** And it is built so
that the answer can be checked — every score decomposes into the metrics behind
it, every fair value shows the methods that produced it and the ones that could
not run, and every number carries a confidence and a provenance trail.

---

## Running it

First time only:

```bash
cd /Users/jspaid/Spaid-Analytics
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
(cd web && npm install && npm run build)
```

Then, in every new terminal, **activate the virtualenv first**. The `spaid`
command is installed inside `.venv`, so without this your shell reports
`command not found: spaid`:

```bash
cd /Users/jspaid/Spaid-Analytics
source .venv/bin/activate

spaid run      # fetch data, then score and value the universe
spaid serve    # http://127.0.0.1:8000
```

Or skip activation and call it by path, which does the same thing:

```bash
.venv/bin/spaid serve
```

Other commands:

```bash
spaid top -n 25       # the ranked list in the terminal
spaid stock NVDA      # one company's full analysis
spaid explain NVDA    # every metric behind the score, with the arithmetic
spaid health          # data-quality report
```

The frontend is a React application in `web/`. `npm run build` produces
`web/dist`, which the API serves at the same origin, so there is one process and
one port. `npm run dev` proxies `/api` to the backend for frontend development.

A full cold run takes about ten minutes, most of it reading SEC filings. A daily
refresh (prices plus rescoring) takes under a minute. The data already in
`data/` is current as of the last run, so `spaid serve` works immediately
without re-running anything.

---

## What it does

```
  providers            canonical tables           derived              interface
  ─────────            ────────────────           ───────              ─────────
  SEC XBRL      ──►    observations      ──►      metrics      ──►     API
  Yahoo prices  ──►    fundamentals      ──►      scores       ──►     React app
  Yahoo quotes  ──►    prices            ──►      valuations   ──►     CLI
  Wikipedia     ──►    companies         ──►      risk
                       securities                 value traps
                       estimates                  confidence
```

**Scoring.** A 0–100 composite from four categories with fixed, versioned
weights: Quality 30%, Growth 25%, Momentum 25%, Valuation 20%. Forty-three
metrics, each percentile-ranked against a defensible peer group — profitability
and valuation within the industry, price behaviour against the whole universe.
The weights are declared in `spaid/config/scoring.py` and are never fitted to
historical returns.

**Fair value.** A range, never a point. Discounted cash flow, comparable
companies and the company's own valuation history, blended 50/30/20 for an
ordinary operating company. Banks get residual income instead of a cash-flow
model, REITs get adjusted funds from operations and net asset value, cyclicals
get normalised mid-cycle earnings. A method that cannot run on the available
data is dropped, the remaining weights renormalise, and the interface says why.

**Confidence.** Scored separately from attractiveness, and deliberately so: a
70 with high confidence and a 70 with low confidence are different objects, and
blending them would destroy the distinction. Six components — data completeness,
freshness, agreement between valuation methods, score stability, business
predictability, analyst coverage.

**Value traps.** Twelve independent signals, because "undervalued" and "worth
buying" are not the same claim. The output classifies the situation: attractively
undervalued, speculatively undervalued, possible value trap, overvalued
high-quality compounder, or overvalued and deteriorating.

---

## The rules the code enforces

These come from the product brief and are not aspirational; each one is a
mechanism somewhere in the code, and most have a test.

**Missing data is never zero.** Every metric carries a status: `scored`,
`missing`, `not_applicable`, or `thin_peers`. A metric with no data is dropped
from its category and the remaining weights renormalise. Below a coverage floor
the category declines to score at all, and below a total-coverage floor so does
the composite. A bank is not penalised for having no gross margin; it is a
metric that does not apply to it.

**Three dates, never one.** Every financial observation records the period it
describes, the date it was filed, and the date we could first have acted on it.
Keying on the period end would hand the model five weeks of foresight on every
fundamental. Every downstream join is an as-of join on availability.

**Weights are policy, not output.** The category weights are fixed and versioned.
Information coefficients are measured and reported as diagnostics, and never fed
back into the weights.

**The interface computes nothing.** The API sends both the number and its
qualitative reading — band, classification label, confidence label, assessment
level and text. The client formats and arranges.

**A company is not a ticker.** Everything joins on a SEC Central Index Key.
Dual-class issuers occupy one row, not two, so a twenty-name portfolio cannot
hold the same business twice.

---

## What the audit of the previous build found

The repository already contained a factor model and a dashboard. Four defects
invalidated its output, all confirmed by direct probe before anything was
rewritten:

| Defect | Effect |
|---|---|
| Information-coefficient gate never opened (max t-statistic 1.89 against a threshold of 2) | Every weight was zero, all 1.1M rows scored exactly 0.0, every stock rated hold |
| Tied scores ranked by unstable sort | The backtest reported 21.3% a year against SPY's 15.1% for an alphabetically-ordered basket |
| Minimum-price screen applied to the split-adjusted close | Nvidia excluded from the investable universe from 2013 to 2017; a future split retroactively rewrote past eligibility |
| Split factor read at the price date, not the filing date | Nvidia's market capitalisation fell from $2.97T to $0.29T for 55 trading days after its 2024 split |

Three further data defects were found and fixed during the rebuild:

- **Share counts.** Dual-class issuers report per class, and the SEC's aggregate
  interface carries only undimensioned facts, so Berkshire's count was frozen at
  941,481 Class A shares from 2011 and its market capitalisation read $477
  million. Twenty-five companies carried decade-stale counts. Now sourced from
  diluted weighted-average shares, with a market-data provider as fallback and a
  cross-check against net income divided by earnings per share.
- **Business classification.** "Asset Management & Custody Banks" contains the
  word "bank", so substring matching valued Ameriprise on its book equity like a
  lender and produced a 74% overvaluation. The financial sector is now
  enumerated explicitly.
- **Scenario fade.** The bull case faded growth *faster* than the base case
  because the fade exponent was inverted, producing bull valuations below base
  for fast-growing companies.

The legacy Dagster prototype is archived under `legacy/`. It could not run in the
current environment and its universe was six hard-coded tickers.

---

## Layout

```
spaid/
  config/       versioned specifications: scoring, valuation, portfolio
  providers/    one adapter per source, all writing the canonical schema
    sec/        XBRL extraction, concept mapping, predecessor linking
    yahoo/      prices, estimates, current quotes
    wikipedia/  index constituents
  storage/      table schemas with validation, and the parquet store
  pipeline/     ingestion, metrics, scoring, risk, confidence, quality checks
  valuation/    discounted cash flow, relative, industry-specific, the blender
  api/          response models, assembly, the HTTP service
web/            React + TypeScript frontend
tests/          unit tests for every financial formula, plus API contract tests
legacy/         the previous prototype, archived unmodified
```

---

## Known limitations

Stated here rather than discovered later.

- **Survivorship bias is half-removed.** Index entry dates prevent trading a
  company before it joined. Deletions are not available from the constituents
  source, so historical results are optimistic by an amount that cannot be
  measured from this data.
- **Analyst estimates have no history.** The source publishes a current snapshot
  only, so estimate history accumulates forward from first collection and the
  backtester must treat it as unavailable at historical dates.
- **Recent spin-offs cannot be assessed on multi-year metrics.** A company with
  two quarters of public financials genuinely has no five-year growth rate.
  Predecessor linking is applied only to reorganisations, where the same business
  continues; attaching a parent's history to a spun-off division would be false.
- **No backtest yet.** The point-in-time backtester, the portfolio engine, and
  the research journal are specified and scaffolded but not built. Until the
  strategy has been validated out of sample against SPY and SPMO, the rankings
  are a structured opinion, not evidence of an edge.

---

## Status

Milestone one is complete: the universe is ranked, every score is explainable to
the individual metric and peer group, fair values carry ranges and confidence,
and the calculations are tested.

Next, in order: the point-in-time backtester; validation against SPY and SPMO;
the portfolio recommendation engine; the Today dashboard; the research journal.
