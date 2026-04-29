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
# # 02 — EDA
#
# Univariate distributions, group-wise PVar means, the empirical heteroskedasticity
# pattern, the right-tail behaviour that motivates the quantile regression.

# %%
from pvar_linreg.plotting import set_style

set_style()

# %% [markdown]
# ## 1. Univariate
# `PVar`, `PVar_abs`, `lead_time_days`, `expected_rate`.

# %% [markdown]
# ## 2. Group-wise PVar means
# By tier x channel - preview of the interaction effect.

# %% [markdown]
# ## 3. Heteroskedasticity preview
# `PVar_abs` variance vs. `expected_rate` and `lead_time_days`. Sets up the diagnostic
# discussion in `03_modeling.ipynb`.

# %% [markdown]
# ## 4. Right-tail
# 95th / 99th / 99.9th percentile of PVar_abs by tier. Sets up the quantile regression.

# %% [markdown]
# ## Findings & next steps
#
# - _Phase 4_
# - _Phase 4_
# - _Phase 4_
