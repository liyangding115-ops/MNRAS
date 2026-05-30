# Magnetic Field Strength as the Dominant Predictor of Pulsar Glitch Regularity

**MNRAS Letter — Analysis pipeline & data release**

This repository contains the complete analysis pipeline, data tables, and 
supplementary material for the MNRAS Letter *"Magnetic Field Strength as 
the Dominant Predictor of Pulsar Glitch Regularity: A Population-Level 
Diagnostic"*.

---

## Overview

We examine the statistical morphology of interglitch waiting times for 
50 pulsars with N_glitch ≥ 4 from the Jodrell Bank and ATNF catalogues. 
The pipeline implements:

- **Coefficient of variation (CV)** computation with measurement-error 
  estimation
- **α-stable distribution fitting** (McCulloch 1986 quantile estimator) 
  and AIC-based model selection against exponential and lognormal 
  alternatives
- **Weighted Spearman correlation** and **weighted multiple regression** 
  (accounting for heterogeneous σ_CV)
- **Multicollinearity diagnostics** (VIF)
- **Leave-one-out robustness checks** and BC_a-corrected bootstrap 
  resampling
- **Figure generation** for the main text and supplementary material

---

## Repository structure
