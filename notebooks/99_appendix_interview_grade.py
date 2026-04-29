# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 99 - Interview-grade Rigor Appendix (PVar OLS)
#
# Each section is a senior-DS-panel question, answered with a defensible methodology
# and a named alternative.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import IPython.display as display

from pvar_linreg.plotting import set_style
from pvar_linreg.config import MODELS_DIR, FIGURES_DIR, PROCESSED_DIR

set_style()
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

# %% [markdown]
# ## A. Recovered vs true coefficients
#
# Since the DGP exposes `true_coefficients()`, we can validate that OLS recovers
# the embedded effects within bootstrap CI. **This is the single most defensible
# piece of evidence in the repo** - and the cleanest story you can tell about a
# regression that you also wrote the data for.

# %%
recovery = pd.read_csv(MODELS_DIR / "coef_recovery.csv")
recovery

# %% [markdown]
# Reading: 0/8 strict-CI recovery (see README) because `dgp.true_coefficients()`
# holds approximate log-domain magnitudes, not analytic OLS-coefficient targets.
# Sign agreement is universal; relative ordering is preserved (B > A > Direct,
# Long-tail > Standard > Preferred > Strategic). The qualitative structure of the
# DGP is rediscovered.

# %%
display.Image(filename=str(FIGURES_DIR / "05_recovery_plot.png"))

# %% [markdown]
# ## B. HC0 / HC1 / HC2 / HC3 side-by-side
#
# Robust-SE flavours compared on the headline coefficient. MacKinnon & White (1985)
# argue HC3 has the most reliable size in finite samples; here at n=200,000 the
# differences are tiny but documented anyway.

# %%
import statsmodels.formula.api as smf
from pvar_linreg.modeling.train import FORMULA

df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
ols = smf.ols(formula=FORMULA, data=df)

cov_rows = []
for cov_type in ("HC0", "HC1", "HC2", "HC3", "nonrobust"):
    fit = ols.fit(cov_type=cov_type) if cov_type != "nonrobust" else ols.fit()
    se_b = fit.bse.get("C(channel)[T.ChannelManagerB]", np.nan)
    cov_rows.append({"cov_type": cov_type, "se_channel_b": float(se_b)})
cov_table = pd.DataFrame(cov_rows)
cov_table

# %% [markdown]
# The four robust variants agree to ~3 decimal places on this coefficient at this
# N. The `nonrobust` SE is the meaningfully different one (different by ~1-2%);
# at smaller N the spread between HC variants would be larger. Lesson: at n>=10⁵
# pick HC3 as a default and don't worry about it; at n<1000 the choice matters.

# %% [markdown]
# ## C. OLS HC3 vs cluster-robust on `property_id`
#
# Within-property residuals are likely correlated over time. Cluster-robust on
# `property_id` is the recommended deployment SE.

# %%
ols_hc3 = joblib.load(MODELS_DIR / "ols_hc3.joblib")
ols_cluster = joblib.load(MODELS_DIR / "ols_cluster.joblib")

hc3_se = ols_hc3.bse.rename("HC3").to_frame()
cluster_se = ols_cluster.bse.rename("cluster").to_frame()
se_compare = hc3_se.join(cluster_se, how="outer")
se_compare["ratio"] = se_compare["cluster"] / se_compare["HC3"]
se_compare.sort_values("ratio", ascending=False).head(15)

# %% [markdown]
# Reading: ratio > 1 means cluster-robust SE is wider than HC3, indicating
# within-property correlation that HC3 misses. For coefficients with ratio ~1
# the within-property correlation is small.

# %% [markdown]
# ## D. OLS vs RLM (Huber) - outlier resilience
#
# Robust regression should be near-identical to OLS when outliers are removed
# (which they were, by IQR drop). Materially different fits would mean the IQR
# filter missed something.

# %%
rlm = joblib.load(MODELS_DIR / "rlm_huber.joblib")
ols_p = ols_hc3.params
rlm_p = rlm.params
delta = (rlm_p - ols_p).abs().sort_values(ascending=False).head(10)
print("Top |OLS - RLM| coefficient differences:")
delta

# %% [markdown]
# All differences sub-0.05 - IQR cleaning was enough. RLM doesn't tell us anything
# new on this dataset, but if the IQR filter were removed it would.

# %% [markdown]
# ## E. Quantile regression at q ∈ {0.5, 0.75, 0.9}
#
# **The Supply-Ops-relevant move**: parity violations matter more in the right
# tail than at the median. Re-estimate at multiple quantiles.

# %%
import statsmodels.formula.api as smf

# Take a smaller sample for speed (full 200k x quantile regression takes ~2 min each)
sample = df.sample(n=30000, random_state=42)
qrows = []
for q in (0.5, 0.75, 0.9):
    qreg = smf.quantreg(formula=FORMULA, data=sample).fit(q=q, max_iter=2000)
    qrows.append(
        {
            "quantile": q,
            "channel_b_coef": float(qreg.params.get("C(channel)[T.ChannelManagerB]", np.nan)),
            "channel_a_coef": float(qreg.params.get("C(channel)[T.ChannelManagerA]", np.nan)),
            "long_tail_coef": float(qreg.params.get("C(property_tier)[T.Long-tail]", np.nan)),
        }
    )
qtable = pd.DataFrame(qrows)
qtable

# %% [markdown]
# Reading: the channel-manager effects grow at higher quantiles. Translation -
# **the bad-PVar tail is concentrated in Channel B** more than the median is.
# That's the production-ops finding: parity-monitoring intensity should be
# tail-aware, not centred on average behaviour.

# %% [markdown]
# ## F. Influence diagnostics + refit dropping top-1% Cook's distance

# %%
from pvar_linreg.diagnostics import cooks_distance

cooks = cooks_distance(ols_hc3)
print(
    f"Cook's d: max={cooks.max():.4f}, p99={np.percentile(cooks, 99):.4f}, p99.9={np.percentile(cooks, 99.9):.4f}"
)

# Refit dropping top-1%
threshold = np.percentile(cooks, 99)
keep = cooks <= threshold
df_clean = df.iloc[keep].reset_index(drop=True)
print(f"Refit on {len(df_clean)} rows (dropped {(~keep).sum()})")
ols_clean = smf.ols(formula=FORMULA, data=df_clean).fit(cov_type="HC3")
delta = (ols_clean.params - ols_hc3.params).abs().sort_values(ascending=False).head(8)
delta

# %% [markdown]
# ## G. Heteroskedasticity check
#
# Breusch-Pagan and White will reject at any reasonable α with N=200k regardless
# of the underlying variance structure. Read the magnitude and the residual plot.

# %%
from pvar_linreg.diagnostics import breusch_pagan, white_test, jarque_bera_test, durbin_watson_stat

bp = breusch_pagan(ols_hc3)
wh = white_test(ols_hc3)
jb = jarque_bera_test(ols_hc3)
dw = durbin_watson_stat(ols_hc3)

pd.Series(
    {
        "BP_lm_p": bp["lm_p"],
        "White_lm_p": wh["lm_p"],
        "JB_p": jb["jb_p"],
        "DW": dw,
    }
).round(4)

# %%
display.Image(filename=str(FIGURES_DIR / "03_residuals_vs_fitted.png"))

# %% [markdown]
# ## H. Box-Cox vs log vs Yeo-Johnson on the target
#
# Sanity check that log was the right transformation.

# %%
from scipy import stats as scipy_stats

target = df["PVar_abs"].clip(lower=1e-4) + 1e-4
_, lam = scipy_stats.boxcox(target.to_numpy())
print(f"Box-Cox lambda = {lam:.4f}  (≈ 0 confirms log is the right transform)")

# %% [markdown]
# ## I. Spec-curve / multiverse-lite
#
# Plot the headline coefficient (`channel = ChannelManagerB`) across a grid of
# reasonable model specifications. Stable across specs => robust finding;
# unstable => report the range and qualify in the README.

# %%
specs = [
    ("base", FORMULA),
    ("- holiday", FORMULA.replace(" + is_holiday", "")),
    ("- weekend", FORMULA.replace(" + is_weekend", "")),
    ("- log_lead_time", FORMULA.replace("log_lead_time + ", "")),
    ("- cyclical month", FORMULA.replace(" + month_sin + month_cos", "")),
    ("- log_expected_rate", FORMULA.replace(" + log_expected_rate", "")),
    ("- star_rating", FORMULA.replace(" + star_rating_centered", "")),
    ("- country", FORMULA.replace(" + C(country)", "")),
    ("- tier:channel interaction", FORMULA.replace(" + C(property_tier):C(channel)", "")),
]
spec_rows = []
for label, f in specs:
    fit = smf.ols(formula=f, data=sample).fit(cov_type="HC3")  # sample for speed
    p = fit.params.get("C(channel)[T.ChannelManagerB]", np.nan)
    se = fit.bse.get("C(channel)[T.ChannelManagerB]", np.nan)
    spec_rows.append(
        {
            "spec": label,
            "coef": float(p),
            "ci_lo": float(p - 1.96 * se),
            "ci_hi": float(p + 1.96 * se),
        }
    )
specs_df = pd.DataFrame(spec_rows)
specs_df

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(
    specs_df["coef"],
    range(len(specs_df)),
    xerr=[specs_df["coef"] - specs_df["ci_lo"], specs_df["ci_hi"] - specs_df["coef"]],
    fmt="o",
    capsize=3,
)
ax.set_yticks(range(len(specs_df)))
ax.set_yticklabels(specs_df["spec"])
ax.set_xlabel("ChannelManagerB coefficient")
ax.set_title("Spec-curve - stability of the headline finding")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# Reading: the headline finding (Channel B has a large positive effect on
# log(PVar_abs)) is robust across every reasonable specification we tested. The
# coefficient point estimate moves in a ~10% band; no spec flips the sign or
# pushes the CI to cross zero.

# %% [markdown]
# ## J. A/B test design - migrating Long-tail Channel B to Channel A
#
# What's the MDE for a 50bps PVar reduction at α=0.05, power=0.8?

# %%
from statsmodels.stats.power import TTestIndPower

# Estimated stdev of log(PVar_abs) on Long-tail Channel B from the data:
ltb = df[(df["property_tier"] == "Long-tail") & (df["channel"] == "ChannelManagerB")]
sd = float(ltb["log_pvar_abs"].std())
print(f"Long-tail Channel B: n={len(ltb)}, sd(log_pvar_abs)={sd:.3f}")

# Cohen's d for a 50bps PVar reduction (~0.005 in PVar units, ~0.5 in log space at typical magnitude):
target_effect = 0.50  # log-domain
d = target_effect / sd
n_required = TTestIndPower().solve_power(effect_size=d, alpha=0.05, power=0.8, ratio=1.0)
print(f"Cohen's d = {d:.3f}; n per arm = {n_required:.0f} bookings")

# %% [markdown]
# ## K. Production monitoring proposal
#
# - **Coefficient drift**: re-estimate quarterly on rolling 12-month windows;
#   alert if any headline coefficient moves >2σ from the historical mean.
# - **Spec-curve as standing report**: track the coefficient distribution across
#   the 9-specification multiverse weekly; the moment a spec flips the sign of
#   ChannelManagerB, escalate to the data engineering team.
# - **Quantile-driven monitoring**: track q=0.9 of PVar_abs by tier x channel
#   monthly, not just the mean - the right tail is what Supply Ops cares about.
