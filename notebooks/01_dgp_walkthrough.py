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
# # 01 — Data-Generating Process Walkthrough
#
# Make the synthetic-vs-real conversation easy: walk through every layer of `dgp.py`,
# show the marginal distributions, show the embedded interactions, and print the dict of
# `true_coefficients()` that the appendix later uses to validate recovery.

# %%
from pvar_linreg.plotting import set_style

set_style()

# %% [markdown]
# ## 1. Property draw
# Tier mix, country mix, star rating, contracted base rate.

# %% [markdown]
# ## 2. Daily panel
# Cross-join with 365 days; weekend / holiday flags; sinusoidal season multiplier.

# %% [markdown]
# ## 3. Lead-time factor
# Non-linear; concentrated in the last 14 days. This is the Supply-Ops-relevant
# non-linearity that the OLS formula linearises via `log_lead_time`.

# %% [markdown]
# ## 4. Channel observations
# Per-tier baseline parity-violation rates: Strategic 0.5%, Preferred 1%, Standard 3%,
# Long-tail 8%. Channel B more error-prone than Channel A. ~1% decimal-point errors.

# %% [markdown]
# ## 5. PVar = (posted - expected) / expected, PVar_abs = |PVar|

# %% [markdown]
# ## 6. true_coefficients()
# Print the dict so a reviewer can see exactly what the regression is supposed to
# recover.

# %% [markdown]
# ## Findings & next steps
#
# - _Phase 4_
# - _Phase 4_
# - _Phase 4_
