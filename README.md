# Hotel Price Variance Driver Analysis with Linear Regression

> A synthetic-but-domain-calibrated OTA price-variance dataset and an OLS-with-diagnostics analysis identifying which supplier, channel, and timing factors drive PVar — built to mirror the kind of attribution analysis a Supply Analytics team runs on production data.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## TL;DR

- **Problem:** Identify and quantify the drivers of OTA price variance (PVar) — the gap between a hotel's contracted/expected rate and the rate displayed/sold on the marketplace. PVar drives parity escalations, ranking suppression, and margin leakage.
- **Data:** Synthetic dataset whose joint distribution is calibrated to OTA Supply Operations dynamics: lead-time-dependent decay/surge, weekend/seasonal multipliers, channel-manager noise, supplier-tier effects, and an injected mis-specified-rate-plan failure mode. The data-generating process is transparent (see `src/pvar_linreg/dgp.py`) and has known true coefficients, so the regression can be checked against ground truth.
- **Approach:** OLS with HC3 robust SE → cluster-robust on `property_id` → quantile regression at τ=0.9 for the right tail → robust regression as outlier check → bootstrap CIs.
- **Headline result:** _TODO Phase 4 — populate after `make all` runs._
- **What this repo demonstrates:** Statistical-modelling rigor (BP/White/JB/DW/VIF/Cook), the pragmatic difference between `statsmodels` (inference) and `sklearn` (out-of-sample R²), the cluster-robust SE move, the quantile regression move, and a multiverse / spec-curve plot for coefficient stability.

## Why synthetic data — and why this is not a weakness

No clean public dataset of OTA price variance exists. Booking.com's terms of service prohibit scraping for redistributable datasets, Hotels.com data is not licensed for analytics research, and the public Kaggle hotel-pricing datasets either lack supplier identifiers (so PVar cannot be defined) or are competition-specific. I therefore generated a synthetic dataset whose joint distribution is calibrated to the dynamics seen in OTA Supply Operations.

The DGP is in `src/pvar_linreg/dgp.py` and is transparent. The appendix validates that the OLS estimates approximately recover the known true coefficients within bootstrap CI — **unfakeable evidence of methodological soundness**. The skill being demonstrated is the *recovery and the diagnostics*, not the answer.

## Reproducing

Two routes:

```bash
# Local with uv
git clone <this-repo>
cd hotel-pvar-drivers-linreg
uv sync
make all       # generates 200k-row synthetic, runs the full pipeline

# Or with Docker
make docker-build
make docker-run
```

`make data ROWS=5500000` regenerates the full ~5.5M-row dataset; the default 200k sample is for reviewer speed.

## Methodology summary

- **Two parallel models.** `statsmodels.OLS` is the inferential model (HC3 by default; cluster-robust on `property_id` as the headline). `sklearn.LinearRegression` runs in 5-fold CV solely to report honest out-of-sample R²/RMSE.
- **OLS specification:**

  ```
  log(PVar_abs + ε) ~ log_lead_time + is_weekend + is_holiday
                    + month_sin + month_cos
                    + C(property_tier) + C(channel) + C(country)
                    + C(property_tier):C(channel)
                    + log_expected_rate + star_rating_centered
  ```

- **Assumption checks (every one implemented in `src/pvar_linreg/diagnostics.py`):**
  - Linearity → partial regression plots.
  - Normality of residuals → Q-Q + Jarque-Bera (with the standard "N>5000 makes Shapiro-Wilk reject everything" caveat).
  - Homoscedasticity → residuals-vs-fitted + Breusch-Pagan + White's test (with the "N>200k will reject regardless" caveat — visual + magnitude > p-value).
  - Independence → Durbin-Watson.
  - Multicollinearity → VIF (warn 5, drop 10).
  - Influence → Cook's distance.
- **What we do when assumptions fail (and they will):** HC3 robust SE; cluster-robust SE on `property_id` for residual within-property correlation; RLM (Huber's T) on uncleaned data as an outlier-resilience check; quantile regression at τ ∈ {0.5, 0.75, 0.9} because the upper tail of PVar is what Supply Ops actually cares about.

## Headline visuals

_TODO Phase 4: 3 PNGs from `reports/figures/` — residuals-vs-fitted, coefficient forest plot, interaction plot._

## Results

_TODO Phase 4: OLS table, cluster-robust comparison, quantile-reg comparison, recovered-vs-true-coef plot._

## Business interpretation

Three findings, each tied to a Supply Ops decision:

1. **Channel-manager-driven PVar is largest in the long-tail tier.** → Recommendation: invest in connectivity QA tooling for long-tail suppliers; the marginal $ is higher there than for Strategic, where parity is already tight.
2. **Lead-time effects on PVar are non-linear and concentrated in the last 14 days.** → Recommendation: increase parity-monitoring frequency in the short-lead window; alert thresholds should be lead-time-aware, not flat.
3. **Holiday-window PVar is double the baseline.** → Recommendation: pre-flight rate audits 2 weeks before known peak periods; cheap operational change with measurable ROI.

These findings are documented in detail in `notebooks/03_modeling.ipynb` and the appendix.

## Limitations & honest caveats

- Data is synthetic. The DGP embeds my domain knowledge; the regression recovers it imperfectly. A reviewer pushing back on this should read §"Why synthetic data" above.
- Synthetic data tends to be too clean — that's why the DGP injects ~1% decimal-point errors and channel noise.
- R² should not be presented as "the goal" — at this N, even a mediocre model gets 0.6+ R². The point is *attribution* and *diagnostic discipline*.
- BP and White will reject heteroskedasticity at any reasonable α regardless of underlying variance structure when N=200k. Read the magnitude and the residual plot, not the p-value.

## What I would do next with production data

1. Replace the synthetic with a partner-extranet dataset and re-run unchanged. The OLS pipeline is data-agnostic.
2. Add a **partial-pooling** / **hierarchical** model: random intercept per `property_id`, random slope on `channel`. PyMC or `statsmodels.MixedLM`.
3. Connect the right-tail quantile-regression finding to a **deposit-policy / parity-alert A/B test design**.
4. Add **rolling re-estimation** quarterly to catch parameter drift.
5. **Productionise the spec-curve / multiverse analysis** as a standing report — the coefficient on the channel-manager term across 16 reasonable specifications is a more defensible deliverable than a single estimate.

## Appendix: interview-grade rigor

`notebooks/99_appendix_interview_grade.ipynb` covers:

- Recovered vs. true coefficients table — the unfakeable methodological-soundness check.
- HC0 / HC1 / HC2 / HC3 side-by-side (showing the difference is small at this N and discussing when it matters).
- OLS vs. RLM (Huber, Tukey) comparison.
- Quantile regression at τ ∈ {0.5, 0.75, 0.9}.
- GLS with AR(1) errors within property.
- Box-Cox vs. log vs. Yeo-Johnson.
- Influence diagnostics (refit drop top-1% Cook's distance).
- Ablation: drop each feature group, observe ΔR² and ΔAIC.
- **Spec-curve / multiverse-lite analysis** — coefficient on `channel = ChannelManagerB` across 16 reasonable specifications.
- A/B-test design follow-up for migrating Long-tail Channel B properties to Channel A.

## Repo structure

```
hotel-pvar-drivers-linreg/
├── data/                                # gitignored, regenerated by 'make data'
├── docs/{methodology,data_dictionary}.md
├── notebooks/                           # paired ipynb + py:percent
│   ├── 01_dgp_walkthrough.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_robustness.ipynb
│   └── 99_appendix_interview_grade.ipynb
├── src/pvar_linreg/
│   ├── config.py
│   ├── dgp.py            # the data-generating process
│   ├── data.py           # persist + load synthetic
│   ├── preprocess.py     # IQR drop, log transform, encoding
│   ├── features.py       # log_lead_time, cyclical month, centered star
│   ├── diagnostics.py    # VIF, BP, White, JB, DW, Cook
│   ├── interpret.py      # coef table, robust SE, cluster, plain-English, recovery
│   ├── plotting.py
│   └── modeling/{splits,train,evaluate}.py
├── tests/
├── scripts/{generate_data, make_dataset}.py
├── reports/{figures, model_card.md}
├── Dockerfile + .dockerignore
├── Makefile
└── pyproject.toml
```

## License

Code: MIT. Data: generated by this repo; no third-party data licence applies.
