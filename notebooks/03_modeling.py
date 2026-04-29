# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Modelling
#
# OLS with HC3 → cluster-robust on `property_id` (the headline) → diagnostic checks → 80/20
# held-out R² for honesty.

# %%
from pvar_linreg.plotting import set_style

set_style()

# %% [markdown]
# ## 1. Fit OLS with HC3 robust SE

# %% [markdown]
# ## 2. Refit with cluster-robust SE on `property_id` (headline)
# Within-property dependence is real; cluster-robust is the right SE choice. Show the
# difference vs. HC3.

# %% [markdown]
# ## 3. Diagnostics
# - Partial-regression plots
# - Q-Q + Jarque-Bera
# - Residuals-vs-fitted + Breusch-Pagan + White
# - Durbin-Watson on date-ordered residuals
# - VIF table
# - Cook's distance + leverage plot

# %% [markdown]
# ## 4. Interaction plot
# `property_tier × channel` with 95% CI bars — the substantively interesting one.

# %% [markdown]
# ## 5. Held-out R² via 80/20 split + 5-fold CV
# Honest out-of-sample numbers.

# %% [markdown]
# ## 6. Bootstrap 95% CIs
# 1000 cluster-bootstrap replicates by `property_id`.

# %% [markdown]
# ## Findings & next steps
#
# - _Phase 4_
# - _Phase 4_
# - _Phase 4_
