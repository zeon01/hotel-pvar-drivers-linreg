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
# # 04 — Robustness Checks
#
# Show coefficients are stable across reasonable alternative specifications: RLM,
# quantile regression, GLS-AR(1), spline lead-time, with and without outlier removal.

# %%
from pvar_linreg.plotting import set_style

set_style()

# %% [markdown]
# ## 1. OLS (HC3) vs. RLM (Huber, Tukey)
# Robust regression on the *un-cleaned* data. Coefficients should be nearly identical
# when outliers are removed; materially different when they're not.

# %% [markdown]
# ## 2. Quantile regression at τ ∈ {0.5, 0.75, 0.9}
# Show that the channel-manager effect is bigger in the right tail than at the median —
# the Supply-Ops-relevant finding.

# %% [markdown]
# ## 3. GLS with AR(1) errors within property
# Variance reduction is small; report it. Justifies sticking with cluster-robust SE.

# %% [markdown]
# ## 4. B-spline on `lead_time` (df=5)
# Model the non-linearity directly; compare to log specification.

# %% [markdown]
# ## Findings & next steps
#
# - _Phase 4_
# - _Phase 4_
# - _Phase 4_
