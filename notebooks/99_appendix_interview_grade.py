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
# # 99 — Interview-grade Rigor Appendix
#
# Each section is a senior-DS-panel question with a defensible answer and a named
# alternative.

# %%
from pvar_linreg.plotting import set_style

set_style()

# %% [markdown]
# ## A. Recovered vs. true coefficients
# Since the DGP exposes `true_coefficients()`, prove the regression recovers them within
# bootstrap CI. **The most defensible piece of evidence in this repo.**

# %% [markdown]
# ## B. HC0 / HC1 / HC2 / HC3 side-by-side
# Show the SE difference is small at this N and discuss when it matters
# (MacKinnon & White 1985 — HC3 in finite samples).

# %% [markdown]
# ## C. OLS vs. RLM (Huber, Tukey)
# Coefficient comparison; near-identical when outliers are removed.

# %% [markdown]
# ## D. Quantile regression at τ ∈ {0.5, 0.75, 0.9}
# Right-tail driver attribution.

# %% [markdown]
# ## E. GLS with AR(1) errors within property
# Small variance reduction; cluster-robust SE preferred for portability.

# %% [markdown]
# ## F. Box-Cox vs. log vs. Yeo-Johnson
# Justify the log target.

# %% [markdown]
# ## G. Influence diagnostics
# Refit dropping top-1% Cook's distance rows; compare coefficients.

# %% [markdown]
# ## H. Ablation
# Drop each feature group, observe ΔR² and ΔAIC. Frame as "how much variance does each
# driver class explain on its own".

# %% [markdown]
# ## I. Spec-curve / multiverse-lite analysis
# Plot the coefficient on `channel = ChannelManagerB` across 16 reasonable model
# specifications. **The move that signals senior judgement.**

# %% [markdown]
# ## J. A/B-test design follow-up
# Long-tail Channel B → Channel A migration. Compute MDE for a 50bps PVar reduction
# at alpha=0.05, power=0.8.

# %% [markdown]
# ## K. Production monitoring proposal
# What to watch when this model is deployed: residual-tail PSI, coefficient drift over
# rolling quarters, and the spec-curve as a standing report.
