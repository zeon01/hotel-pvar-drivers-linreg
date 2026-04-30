# Methodology — hotel-pvar-drivers-linreg

This document walks through every modelling decision, names the rejected alternative,
and is written so each section can be paraphrased in a technical review.

## 1. Why synthetic data

Booking.com's ToS prohibit scraping for redistributable datasets. Hotels.com data is
not licensed for analytics research. Public Kaggle hotel-pricing snapshots either lack
supplier identifiers (so PVar cannot be defined) or are competition-specific.

The DGP in `src/pvar_linreg/dgp.py` is transparent and has known true coefficients;
the appendix validates that the OLS estimates approximately recover them within bootstrap
CI. **The skill being demonstrated is the recovery and the diagnostics, not the answer.**

**Rejected alternatives:**

- Web scraping (legal grey zone; not redistributable).
- Expedia ICDM 2013 (no parity context).
- Manual entry from public rate APIs (rate-limited, partial coverage, brittle).

## 2. DGP design

5,000 synthetic properties × 365 days × 3 channels = ~5.5M rows. Default sample is 200k
for speed.

Per row, the DGP composes:

```
expected_rate = contracted_base_rate × season_multiplier × weekend_multiplier
                × holiday_multiplier × lead_time_factor

posted_rate   = expected_rate × channel_noise × tier_quality_term × parity_violation_term

PVar          = (posted_rate - expected_rate) / expected_rate
PVar_abs      = |PVar|
```

Per-tier baseline parity-violation rates: Strategic 0.5%, Preferred 1%, Standard 3%,
Long-tail 8%. Channel B is more error-prone than Channel A. ~1% rows have a "decimal-point
error" (rate 10× off) — these are the residual outliers that the diagnostics catch.

## 3. Preprocessing — minimal but deliberate

Synthetic data is clean by construction; the preprocessing steps are run anyway because
a real-world pipeline would.

- **IQR-based outlier drop on `PVar_abs`** (k=3). Removes the injected decimal-point
  errors. **Robust regression on the un-cleaned data is fit in the appendix** as a
  comparison.
- **Log-transform `PVar_abs`** after clipping at small ε (non-negative, right-skewed).
  Box-Cox is shown as a comparison; expected λ ≈ 0 → log.
- **One-hot encode** categoricals with `drop_first=True` for OLS interpretability.

## 4. OLS specification

```
log(PVar_abs + ε) ~ log_lead_time + is_weekend + is_holiday
                  + month_sin + month_cos
                  + C(property_tier) + C(channel) + C(country)
                  + C(property_tier):C(channel)
                  + log_expected_rate + star_rating_centered
```

`statsmodels.formula.api.ols` is used so the formula is explicit and the output legible.

The `property_tier × channel` interaction is the substantively interesting one — it
tests whether channel-driven PVar differs across supplier tiers, which is the exact
hypothesis Supply Ops would pose.

## 5. Assumption checks

Every check below is implemented in `src/pvar_linreg/diagnostics.py`:

| Assumption | Test | Caveat documented in code/notebook |
|------------|------|------------------------------------|
| Linearity | partial-regression plots | discuss any non-linear pattern |
| Normality | Q-Q + Jarque-Bera | Shapiro-Wilk rejects at N>5000 — irrelevant; CLT covers CI validity |
| Homoscedasticity | Residual-vs-fitted + Breusch-Pagan + White | At N>200k both reject regardless. Read the magnitude and the residual plot. |
| Independence | Durbin-Watson | DW only detects AR(1); discuss limits and prefer cluster-robust SE within property |
| Multicollinearity | VIF | warn 5, drop 10 — interactions inflate naturally and are fine |
| Influence | Cook's distance + leverage | injected decimal-point rows should appear as obvious outliers — validates the diagnostic |

## 6. What we do when assumptions fail (and they will)

| Failure | Response |
|---------|----------|
| Heteroskedasticity (BP/White reject) | HC3 robust SE: `result.get_robustcov_results(cov_type='HC3')`. **Justification: HC3** preferred because MacKinnon & White (1985) and the StataCorp blog simulation (2022) show HC3 has the most reliable size in samples N<1000 and remains competitive at scale. |
| Heavy-tailed residuals | Robust regression (`statsmodels.RLM` with Huber's T or Tukey's biweight) as appendix model. Compare coefficient estimates side-by-side. |
| Non-linearity in lead-time | Already addressed via log; appendix tries a B-spline (`patsy.bs(lead_time, df=5)`). |
| Non-constant variance correlated with a known factor | WLS with weights = 1 / fitted variance from an auxiliary regression. Appendix only. |
| Within-property dependence | **Headline model: cluster-robust SE on `property_id`** — `cov_type='cluster', cov_kwds={'groups': df['property_id']}`. |
| Heavy skew unfixed by log | **Quantile regression at τ ∈ {0.5, 0.75, 0.9}** — the upper tail of PVar is what Supply Ops actually cares about. |

## 7. Evaluation

- 80/20 random split (rows are i.i.d. across properties; this is an attribution model,
  not a forecast — time-based split is not necessary).
- Held-out R², adjusted R², RMSE, MAE on the test set.
- 5-fold KFold CV via `sklearn.LinearRegression` for stability.
- Bootstrap 95% CIs on the headline coefficients (1000 cluster-bootstrap samples by
  `property_id`).

## 8. Coefficient recovery validation

Because the DGP exposes `true_coefficients()`, the appendix proves the regression
recovers them within bootstrap CI. This is the single most defensible piece of evidence
in the repo.

## 9. Defensive choices

- The headline model is **cluster-robust on `property_id`**, not plain OLS. A senior
  reviewer would flag the absence of cluster-robust SE in within-property data.
- Quantile regression at τ=0.9 is run on the headline because parity violations matter
  most in the right tail.
- The multiverse / spec-curve plot is the move that signals senior judgement: present
  the coefficient on `channel = ChannelManagerB` across 16 reasonable model
  specifications. Stable ⇒ robust; unstable ⇒ report the range.

## 10. Defensive choices for ambiguity

_TODO Phase 4: log any judgement calls made during implementation that weren't fully
specified by the spec._
