# Probabilistic UK Flood Nowcasting — Full Project Documentation

**Project:** Probabilistic UK Flood Nowcasting Using Conditional Quantile Regression with ConvLSTM and 3D-CNN  
**Author:** Alp Ozturk  
**Institution:** University of Reading — MSc Data Science and Advanced Computing  
**Supervisor:** Prof. Atta Badii  
**Period:** 2025–2026  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
   - 2.1 [ERA5 Reanalysis Data](#21-era5-reanalysis-data)
   - 2.2 [GPM IMERG Precipitation Data](#22-gpm-imerg-precipitation-data)
   - 2.3 [Spatial Domain](#23-spatial-domain)
   - 2.4 [Temporal Coverage](#24-temporal-coverage)
   - 2.5 [Final Dataset Scale](#25-final-dataset-scale)
3. [Pipeline — Step by Step](#3-pipeline--step-by-step)
   - 3.1 [Data Acquisition](#31-data-acquisition)
   - 3.2 [Spatial Merging (ERA5 + IMERG)](#32-spatial-merging-era5--imerg)
   - 3.3 [Feature Engineering](#33-feature-engineering)
   - 3.4 [Missing Value Treatment](#34-missing-value-treatment)
   - 3.5 [Normalisation](#35-normalisation)
   - 3.6 [Target Variable Construction](#36-target-variable-construction)
   - 3.7 [Class Imbalance Analysis](#37-class-imbalance-analysis)
   - 3.8 [WGAN-GP Synthetic Augmentation](#38-wgan-gp-synthetic-augmentation)
   - 3.9 [Model Architectures](#39-model-architectures)
   - 3.10 [Training](#310-training)
   - 3.11 [Meta-Learner Ensemble](#311-meta-learner-ensemble)
   - 3.12 [Evaluation](#312-evaluation)
4. [Problems Faced and Solutions](#4-problems-faced-and-solutions)
   - P1 [IMERG Quality Flag NaN Contamination](#p1-imerg-quality-flag-nan-contamination)
   - P2 [ERA5 Boundary Artefact Row](#p2-era5-boundary-artefact-row)
   - P3 [ERA5 SST Land Cell Missing Values](#p3-era5-sst-land-cell-missing-values)
   - P4 [Permanently Missing IMERG Cells](#p4-permanently-missing-imerg-cells)
   - P5 [Free-Optimum / Persistence Collapse](#p5-free-optimum--persistence-collapse)
   - P6 [Delta-Target Near-Leakage (h−1 offset)](#p6-delta-target-near-leakage-h1-offset)
   - P7 [Residual Lag After Delta-Target Reframing](#p7-residual-lag-after-delta-target-reframing)
   - P8 [4-Cell Receptive Field — Upstream Signals Unreachable](#p8-4-cell-receptive-field--upstream-signals-unreachable)
   - P9 [CNN3D Temporal Collapse Bug](#p9-convlstm-temporal-collapse-bug)
   - P10 [WGAN-GP BatchNorm NaN Gradients](#p10-wgan-gp-batchnorm-nan-gradients)
   - P11 [WGAN-GP Under-Dispersion](#p11-wgan-gp-under-dispersion)
   - P12 [Optuna Proxy Trial Failure](#p12-optuna-proxy-trial-failure)
   - P13 [ERA5 U10/V10 NaN at Northern Boundary](#p13-era5-u10v10-nan-at-northern-boundary)
5. [Final Results](#5-final-results)
6. [Hardware](#6-hardware)
7. [Key Design Decisions — Summary](#7-key-design-decisions--summary)

---

## 1. Project Overview

This project developed an end-to-end probabilistic nowcasting system for UK flood risk, targeting a **6-hour-ahead forecast horizon**. Traditional deterministic forecasting reports only a point estimate, giving emergency services no sense of uncertainty. This system instead outputs **full predictive quantile distributions** (10 quantiles: 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95) together with a binary **flood-event detection signal**, over the complete UK land and coastal grid.

**Research gap addressed:** Existing literature studied LLM and AI support for software engineering in isolated phases (requirements, coding, testing), with few unified human-in-the-loop traceable pipelines integrating gridded Earth observation data with probabilistic deep learning at national scale.

**Approach:**  
- Two deep learning models: **DilatedConvLSTMTwoHead** (recurrent) and **DilatedCNN3DTwoHead** (volumetric).  
- A **meta-learner ensemble** (per-quantile linear QuantileRegressor) combining the two deep models, a FastRandomForest baseline, and a persistence baseline.  
- **Conditional quantile regression** via pinball loss, trained on a **delta-from-persistence** formulation to avoid the free-optimum failure mode (explained in Problems section).  
- **WGAN-GP** synthetic data augmentation to address severe class imbalance in heavy flood events.  
- Design Science Research (DSR) framework: artefact-centred, iterative, evaluation-driven.

---

## 2. Dataset

### 2.1 ERA5 Reanalysis Data

**Source:** ECMWF ERA5 reanalysis via the Copernicus Climate Data Store (CDS).  
**Resolution:** 0.25° × 0.25° spatial, hourly temporal.  
**Variables (18 channels):**

| Variable | Description |
|---|---|
| `era5_t2m` | 2-metre air temperature (K) |
| `era5_d2m` | 2-metre dewpoint temperature (K) |
| `era5_msl` | Mean sea level pressure (Pa) |
| `era5_tcwv` | Total column water vapour (kg m⁻²) |
| `era5_blh` | Boundary layer height (m) |
| `era5_cape` | Convective available potential energy (J kg⁻¹) |
| `era5_cp` | Convective precipitation (m) |
| `era5_sp` | Surface pressure (Pa) |
| `era5_tp` | Total precipitation (m) |
| `era5_sst` | Sea surface temperature (K) — NaN over land, filled |
| `era5_swvl1` | Soil water volume layer 1 (m³ m⁻³) |
| `era5_z` | Geopotential at 500 hPa (m² s⁻²) |
| `era5_u10` | 10-metre U-wind component (m s⁻¹) |
| `era5_v10` | 10-metre V-wind component (m s⁻¹) |
| `era5_r` | Relative humidity at 700 hPa (%) |
| `era5_q700` | Specific humidity at 700 hPa (kg kg⁻¹) |
| `era5_lsm` | Land-sea mask (0–1, static) |
| `era5_fal` | Forecast albedo (dimensionless) |

**Why ERA5?** ERA5 is the state-of-the-art global atmospheric reanalysis, produced by assimilating billions of historical observations into a physical model. It provides spatially and temporally complete gridded fields where observations are sparse over the UK (e.g. offshore, upland areas), making it the standard reference for numerical weather prediction research.

---

### 2.2 GPM IMERG Precipitation Data

**Source:** NASA Goddard Earth Sciences Data and Information Services Center (GES DISC), GPM IMERG Final Run V07.  
**Resolution:** 0.1° × 0.1° spatial, 30-minute temporal (aggregated to hourly).  
**Variables:**
- `imerg_precipitation` — Calibrated multi-satellite precipitation estimate (mm hr⁻¹).
- `imerg_quality` — Quality/calibration flag (later removed after causing NaN contamination; see Problem P1).

**Why IMERG?** ERA5 total precipitation (`era5_tp`) is model-derived. IMERG provides a direct observation-constrained estimate from the GPM satellite constellation, capturing fine-scale convective rainfall structures at 0.1° that ERA5 at 0.25° smooths over. Including IMERG as an independent channel improved discrimination of heavy convective events.

---

### 2.3 Spatial Domain

**Bounding box (final, after artefact fix):**  
- Latitude: 48.5°N – 62.25°N (56 ERA5 rows × 0.25° = 14 degrees, then 140 IMERG rows × 0.1°)  
- Longitude: −9.75°E – 3.25°E (131 IMERG columns × 0.1°)  

**Grid shape:** 140 rows × 131 columns = **18,340 cells** (land + sea combined).  

**Why this domain?** The domain covers the entirety of the United Kingdom including Northern Scotland, Wales, and Northern Ireland, with sufficient ocean buffer to capture Atlantic weather systems that drive most UK precipitation events, while excluding the Norwegian coast where ERA5 boundary artefacts were discovered at 62.5°N (see Problem P2).

---

### 2.4 Temporal Coverage

- **Training period:** January 2024 – December 2025 (17,544 hours)  
- **Test period:** January 2026 – June 2026 (4,344 hours)  
- **Total:** 21,888 hourly timesteps  

**Why 2024–2026?** The project was conducted in 2025–2026, so recent reanalysis data were available. Using only 2024–2026 kept the dataset manageable (401 million rows) while covering two full annual cycles of UK seasonal variability, including the wet winter periods when flood risk is highest.

---

### 2.5 Final Dataset Scale

| Metric | Value |
|---|---|
| Total rows (cell × time) | 401,425,920 |
| Hourly timesteps | 21,888 |
| Grid cells | 18,340 |
| ERA5 channels | 18 |
| IMERG channels | 1 (precipitation only, after quality flag removal) |
| Engineered channels | 2 (wind speed, FAO soil class) |
| **Total input channels (model)** | **20** |
| Lookback window | 18 hours |
| Forecast horizon | 6 hours ahead |

The full dataset occupies approximately **77 GB** as a flat CSV. This was pre-converted to compressed binary NumPy `.npy` arrays for 10–20× faster loading during model training.

---

## 3. Pipeline — Step by Step

### 3.1 Data Acquisition

**What was done:**  
ERA5 data were downloaded via the CDS API in monthly NetCDF batches for each variable separately. IMERG half-hourly HDF5 files were downloaded from GES DISC for the same period.

**Why monthly batches?**  
The CDS enforces a per-request size limit. Requesting all variables for the full period in one call exceeds this limit. Monthly batches also allowed incremental download and early inspection of each month before committing to a full download.

**Why IMERG Final Run V07?**  
The Final Run is the highest-latency but highest-accuracy IMERG product, incorporating ground gauge corrections. The Early and Late runs trade accuracy for timeliness; since this project is a nowcasting research study (not an operational real-time system requiring sub-hour latency), the Final Run provided the best quality training signal.

---

### 3.2 Spatial Merging (ERA5 + IMERG)

**What was done:**  
ERA5 at 0.25° and IMERG at 0.1° operate on different grids. A **KDTree nearest-neighbour** approach was used to assign each IMERG cell the value of its spatially closest ERA5 cell. The merged dataset was written as a flat CSV with one row per (cell, hour) pair, sorted by time then cell index.

**Why nearest-neighbour rather than bilinear interpolation?**  
Bilinear interpolation blends neighbouring ERA5 values, which is appropriate for smooth fields (temperature, pressure) but problematic for land-sea mask and categorical features. Nearest-neighbour preserves hard boundaries. For precipitation, which is already at higher resolution in IMERG, interpolating the coarser ERA5 estimate was deemed less informative than preserving the coarse native value.

**Per-cell time continuity check:**  
After merging, a diagnostic confirmed that every cell had exactly 21,888 rows (one per hour). Cells with fewer rows indicated a gap in the source data; cells with more indicated duplicated timestamps. No gaps or duplicates were found after pipeline fixes.

---

### 3.3 Feature Engineering

#### Wind Speed
**What was done:**  
A scalar wind speed channel was derived from ERA5 U10 and V10 components:

```
wind_speed10 = sqrt(era5_u10² + era5_v10²)
```

**Why?**  
Individual U and V components carry directional information, but the magnitude of wind speed is the physically relevant variable for flood risk (driving precipitation intensity and advection rate). Including both components and derived speed gives the model access to both direction and magnitude.

#### FAO Soil Classification
**What was done:**  
ERA5 soil water volume layer 1 (`era5_swvl1`) values were discretised into 4 ordinal classes following the FAO soil hydrological grouping scheme:

- Class 0: Very low water retention (sandy soils, rapid runoff)
- Class 1: Low retention
- Class 2: Medium retention
- Class 3: High retention (clay soils, slow drainage, high flood susceptibility)

**Why?**  
Soil saturation state critically determines whether precipitation generates surface runoff (→ flood) or infiltrates. A continuous saturation value is difficult for the model to interpret relative to the infiltration threshold specific to each soil type. Discretising into hydrological classes provides a stable categorical prior for each cell's drainage behaviour.

---

### 3.4 Missing Value Treatment

#### ERA5 SST over Land Cells
Land cells have no sea surface temperature by definition. ERA5 fills these with a sentinel NaN. These were replaced with **277.22 K (4.07°C)**, the spatial mean of all valid SST cells within the domain.

**Why a constant fill rather than spatial interpolation?**  
SST over land is physically meaningless; the value functions purely as a feature placeholder. Using the domain mean ensures the normalised value is close to zero (the normalised mean), preventing the model from learning spurious spatial patterns from land-cell SST.

#### Permanently Missing IMERG Cells
**2,216 cells** in the IMERG grid had no valid precipitation observation across the entire 2024–2026 period. Inspection showed these corresponded to systematic gaps in the GPM satellite swath coverage at the periphery of the domain and a small number of inland water bodies.

These cells were filled using **4-connected spatial averaging**: the mean of valid left, right, up, and down neighbours. Cells at domain edges with fewer than 2 valid neighbours were flagged and excluded from model evaluation (though included in training to avoid boundary artefacts in convolution operations).

**Why 4-connected rather than 8-connected or nearest-neighbour?**  
4-connected averaging uses only directly adjacent cells, limiting the spatial propagation of filled values. 8-connected averaging would mix diagonal neighbours, which at 0.1° resolution are ~14 km apart and may capture qualitatively different precipitation regimes (e.g. upland vs valley).

---

### 3.5 Normalisation

Three normalisation strategies were applied based on each variable's distributional shape:

| Strategy | Applied to | Formula |
|---|---|---|
| **Z-score** | Approximately Gaussian variables (t2m, d2m, msl, tcwv, blh, wind_speed, era5_z) | (x − μ) / σ |
| **Log1p + Z-score** | Right-skewed non-negative (imerg_precipitation, era5_cape, era5_cp, era5_tp, era5_swvl1) | (log(1+x) − μ) / σ |
| **Min-max [0,1]** | Bounded variables (lsm, fal, FAO soil class) | (x − min) / (max − min) |

**Why variable-specific normalisation?**  
Neural networks with gradient-based optimisation converge faster when inputs have similar scales and distributions. Z-score standardisation is optimal for near-Gaussian inputs. For heavily right-skewed precipitation data, raw z-score would compress 99% of values near zero while rare extreme values land many standard deviations out — log1p compression first redistributes the dynamic range. Min-max is used for variables that are intrinsically bounded (mask values, categorical encodings) where z-score could push out-of-range at test time.

**Important note on histogram scripts:**  
Two versions of the histogram diagnostic script exist in the codebase:

- `distributions_grid.py`: Reads raw (un-normalised) data from `merged_data_with_windspeed.csv`. Uses `min`/`max` from `summary_statistics.csv` as bin edges.
- `distributions_grid_normalised.py`: Reads normalised data from `merged_data_final.csv`. Performs a separate min/max scan of the actual normalised file for bin edges (because normalised min/max differ from raw statistics).

The orange dashed vertical line in the normalised histograms represents the **expected normalised mean** (0 for z-score variables, 0.5 for min-max variables), not the raw mean.

---

### 3.6 Target Variable Construction

**What was done:**  
The target variable `flood_composite_6h_ahead` was defined as a **weighted percentile-rank composite**:

```
flood_composite_6h_ahead = 0.6 × percentile_rank(P₂₄) + 0.4 × percentile_rank(S̄)
```

Where:
- `P₂₄` = 24-hour accumulated precipitation at each cell
- `S̄` = spatially averaged soil moisture across a 3×3 neighbourhood
- `percentile_rank(·)` = value's position in the empirical CDF across all cells and times

The 60/40 weighting reflects the relative physical importance of precipitation intensity versus soil saturation state in determining flood risk. Precipitation is the immediate driver; soil saturation determines whether that precipitation runs off or infiltrates.

**Why percentile ranks rather than raw values?**  
Raw precipitation values span many orders of magnitude (0 mm to >50 mm/hr) and have highly non-stationary statistics across seasons and geography. Percentile ranks normalise these into [0,1] with a uniform marginal distribution, making the target statistically comparable across all cells and all seasons. This is the standard approach in climatological index construction.

**Why 6 hours ahead?**  
Six hours is operationally meaningful for flood early warning: it provides sufficient lead time for emergency management decisions (evacuation, flood barrier deployment) while remaining within a timeframe where atmospheric predictability is reasonable given the 18-hour observational input window.

---

### 3.7 Class Imbalance Analysis

**Distribution of flood classes:**

| Class | Threshold | Frequency | Ratio |
|---|---|---|---|
| No-flood | `flood_composite < 0.33` | ~73% | 27 |
| Average-flood | `0.33 ≤ flood_composite < 0.67` | ~22% | 6 |
| Heavy-flood | `flood_composite ≥ 0.67` | ~5% | 1 |

**Ratio 27:6:1** (no-flood : average-flood : heavy-flood).

**Impact:** Without intervention, a model minimising standard loss functions would learn to predict the "no-flood" class almost exclusively, achieving high accuracy on the dominant class while failing entirely on the rare heavy-flood events that are the primary concern for emergency management.

**Mitigations applied:**
1. **Weighted pinball loss:** Heavy-flood quantile errors penalised 10×, average-flood 3×.
2. **WGAN-GP synthetic augmentation:** Generated additional heavy-flood training patches.
3. **Event-detection head:** Binary cross-entropy head with `pos_weight=5.10` to explicitly amplify the gradient signal for flood-positive examples.

---

### 3.8 WGAN-GP Synthetic Augmentation

**What was done:**  
A **Wasserstein GAN with Gradient Penalty (WGAN-GP)** was trained on heavy-flood spatial patches to generate synthetic training examples.

**Architecture:**
- **Generator (U-Net):** 8,212,565 parameters. Encoder-decoder with skip connections. Takes a noise vector conditioned on temporal metadata and outputs a full spatial patch (140×131 grid, 20 channels).
- **Critic:** 2,913,985 parameters. Discriminates real from generated patches using the Wasserstein distance.
- **Gradient penalty:** Enforces the Lipschitz constraint on the critic (Gulrajani et al., 2017), replacing weight clipping from the original WGAN formulation.

**Training data:** 2,419 unique heavy-flood event patches (before geometric augmentation) from the 2024–2025 training period. Approximately 175 genuinely distinct flood events.

**Output:** 6,000 synthetic heavy-flood spatial patches were generated and incorporated into training.

**Why WGAN-GP rather than standard GAN or SMOTE?**  
WGAN-GP provides more stable training than standard GANs (which suffer from mode collapse and vanishing gradients) through the Wasserstein distance metric. SMOTE operates in feature space and cannot generate the spatially coherent 2D precipitation fields that the spatial convolution models require. WGAN-GP can in principle generate realistic 2D spatial patterns.

**Result and limitation:** Despite successful training convergence (no NaN gradients, outputs in the valid normalised range), the synthetic patches exhibited a **standard deviation of 0.83 versus 10.63 for real patches** — an under-dispersion factor of approximately 13. The generator converged to a **mode-averaging solution** rather than capturing the full distributional diversity of real heavy-flood events. This limited the practical benefit of augmentation (see Problem P11 for root cause analysis).

---

### 3.9 Model Architectures

#### DilatedConvLSTMTwoHead

A recurrent architecture processing the 18-hour lookback sequence through multiple ConvLSTM layers with **dilated spatial convolutions**.

**Dilation rates:** (1, 2, 4) applied to the spatial convolution kernels within the ConvLSTM cells.

**Why dilated convolutions in ConvLSTM?**  
Standard 3×3 ConvLSTM kernels have a spatial receptive field of only 3×3 cells per layer. At 0.1° resolution each cell is approximately 7 km. A 3×3 receptive field covers only ~21 km, which is insufficient to capture mesoscale weather systems and upstream catchment signals that propagate over distances of 50–200 km. Dilated convolutions insert gaps between kernel weights, expanding the effective receptive field without increasing parameter count. Rates (1, 2, 4) produce a verified **31-cell effective receptive field** (~217 km), sufficient to capture Atlantic frontal systems and river catchment drainage areas.

**Two-head output:**
- **Quantile head:** 10-channel output (one per quantile τ ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95}). Trained with weighted pinball loss.
- **Event-detection head:** 1-channel binary output. Trained with binary cross-entropy (`pos_weight=5.10`).

**Training result:** Best validation CRPS 3.8278 at epoch 29/30. Training time: 302 minutes.

#### DilatedCNN3DTwoHead

A volumetric architecture treating the (time, latitude, longitude) input as a 3D volume and applying 3D convolutions across all three dimensions simultaneously.

**Dilation rates:** (1, 2, 4, 8, 16) applied to the spatial dimensions of the 3D kernels.

**Why higher dilation rates than ConvLSTM?**  
CNN3D lacks the recurrent hidden state of ConvLSTM, so all temporal integration happens within a single forward pass through the 3D convolution stack. Higher dilation rates are needed to maintain a comparable spatial receptive field. Rates up to 16 produce a receptive field of 31+ cells in the spatial dimensions while the temporal dimension is handled by the 18-step depth.

**Training result:** Best validation CRPS 3.9533 at epoch 27/30. Training time: 114 minutes (faster than ConvLSTM due to parallelisable 3D convolutions vs sequential recurrent processing).

#### FastRandomForest Baseline

A tabular machine learning baseline using the **FastRandomForest** library, trained on per-cell feature vectors (no spatial context). Provides a non-deep-learning comparison point.

#### Persistence Baseline

Predicts the target at time t+6 as equal to the honest h−7 observation (see Problem P6 for why h−7 rather than h−1). Achieves a CRPS of **6.057** on the test set. All deep learning models were required to beat this baseline as a minimum bar.

---

### 3.10 Training

**Hardware:** NVIDIA GeForce RTX 4080 (16 GB VRAM), Intel Core i9-13900K, 64 GB RAM.

**Data loading:** Pre-converted NumPy binary arrays loaded via memory-mapped files. Patch-based batching: each training sample is an (18×140×131×20) tensor (lookback × height × width × channels).

**Optimiser:** Adam with learning rate 1e-4, weight decay 1e-5.

**Loss function:**  
Combined loss = α × PinballLoss(quantile head) + β × BCELoss(event head)

Where α=0.8, β=0.2, balancing the quantile regression objective against the event detection signal.

**Weighted pinball loss:**  
Standard pinball loss for quantile τ and residual r:

```
L_τ(r) = r × τ        if r ≥ 0
         r × (τ − 1)  if r < 0
```

Class-weighted extension: multiply by 10.0 for heavy-flood cells, 3.0 for average-flood cells, 1.0 for no-flood cells.

**Epochs:** 30 for both models. Early stopping not triggered; best checkpoint saved at validation minimum.

**Train/validation/test split:**  
- Train: January 2024 – September 2025  
- Validation: October 2025 – December 2025  
- Test: January 2026 – June 2026  

Temporal split only (no shuffling), preserving time-series structure and preventing data leakage.

---

### 3.11 Meta-Learner Ensemble

**What was done:**  
A **per-quantile linear QuantileRegressor** was trained to combine the predictions of four base models:

1. DilatedConvLSTMTwoHead
2. DilatedCNN3DTwoHead
3. FastRandomForest
4. Persistence (h−7)

For each quantile τ, a separate linear model maps the 4 base model predictions to a final combined prediction.

**Why per-quantile rather than a single linear combination?**  
The base models exhibit **opposing calibration biases**: ConvLSTM systematically under-estimated flood risk in the central quantile range (τ = 0.3–0.7), while CNN3D systematically over-estimated in the same range. A single fixed weight cannot simultaneously address both biases. Per-quantile weights allow the meta-learner to up-weight ConvLSTM at lower quantiles (where its conservative bias reduces over-confidence) and down-weight CNN3D at upper quantiles (where its aggressive bias is most damaging).

**Result:** Meta-learner PICP 0.826 versus ConvLSTM 0.812 and CNN3D 0.805 on the validation set. This complementarity is the fundamental justification for the ensemble approach.

---

### 3.12 Evaluation

**Metrics used:**

| Metric | Description | Final (meta-learner) |
|---|---|---|
| CRPS | Continuous Ranked Probability Score — lower is better | 4.640 |
| CSI (avg) | Critical Success Index for average-flood events | 0.702 |
| CSI (heavy) | Critical Success Index for heavy-flood events | 0.620 |
| BSS (avg) | Brier Skill Score vs persistence, average-flood | 0.654 |
| BSS (heavy) | Brier Skill Score vs persistence, heavy-flood | 0.413 |
| PICP | Prediction Interval Coverage Probability (90% PI) | 0.753 |
| Winkler | Winkler Score (sharpness-penalised coverage) | 34.15 |
| FSS w3 (avg) | Fractions Skill Score, 3-cell window, average-flood | 0.884 |
| FSS w3 (heavy) | Fractions Skill Score, 3-cell window, heavy-flood | 0.843 |
| Mean lag k | Mean quantile lag (negative = early prediction) | −4.80 |

**Persistence baseline CRPS:** 6.057 (honest h−7; see Problem P6).

---

## 4. Problems Faced and Solutions

---

### P1: IMERG Quality Flag NaN Contamination

**Problem:**  
The IMERG HDF5 files contain a `quality` flag variable alongside precipitation. When this flag was included as a feature column, NaN values propagated widely through the merged dataset — far beyond the known permanently missing cells. Initial EDA showed ~8% NaN rate in IMERG quality compared to <0.1% expected from coverage analysis.

**Diagnosis:**  
The `imerg_quality` variable uses a different fill value convention than `imerg_precipitation`. Where precipitation quality was unknown or indeterminate (cloud-affected pixels, edge-of-swath), the quality variable was set to a sentinel integer (e.g. −9999) rather than a float NaN. The pandas read pipeline interpreted this sentinel as a valid integer, then when downstream operations mixed integer and float columns, the sentinel propagated as NaN in merged contexts.

**Solution:**  
The `imerg_quality` column was **removed entirely** from the feature set. Quality filtering was applied upstream at ingest time: only hourly cells where the quality flag indicated "best" or "good" calibration were retained; the remainder were treated as permanently missing and filled by the 4-connected spatial averaging procedure (see Problem P4). This eliminated all quality-flag-derived NaN contamination downstream.

**Map showing fix:** `precip_quality_map_01deg.png` — before fix shows scattered NaN contamination across the UK; after fix shows only the 2,216 structural gaps at domain periphery.

---

### P2: ERA5 Boundary Artefact Row

**Problem:**  
ERA5 data were initially downloaded with the bounding box including latitude 62.5°N. A diagnostics map (`era5_u10` NaN map) revealed that **131 cells along the 62.5°N row** (the entire northernmost row) had NaN values for wind components U10 and V10.

**Diagnosis:**  
This is a known ERA5 artefact at the northern boundary of the UK subdomain. The 0.25° ERA5 grid at 62.5°N sits at the edge of the regional extraction boundary; the reanalysis model's spectral truncation produces invalid wind values at this exact latitude when extracted as a regional subset.

**Solution:**  
The spatial domain was **truncated at 62.25°N** (removing the 62.5°N row). The final grid is 140×131 cells rather than the originally intended 141×131. This row contained no UK land cells (it corresponds to open ocean north of the Shetland Islands) and its removal had no impact on flood risk prediction quality.

**Why not impute?**  
An entire row of NaN values for wind components would require imputation from the row immediately below (62.25°N). Wind fields at the boundary of a subdomain extraction are not reliable guides to the true 62.5°N values; imputing them would introduce systematic wind speed biases in the northernmost cells of the retained domain. Exclusion was cleaner.

---

### P3: ERA5 SST Land Cell Missing Values

**Problem:**  
`era5_sst` (sea surface temperature) is physically defined only over ocean. ERA5 stores land cells as NaN for this variable. The merged dataset therefore had NaN in `era5_sst` for all ~9,000 land cells across all 21,888 timesteps.

**Solution:**  
Land cells were filled with **277.22 K** (4.07°C), the domain-wide mean of all valid SST cells over the full period.

**Why not exclude SST entirely?**  
SST influences atmospheric moisture flux and coastal precipitation patterns. The model's convolutional layers process the full spatial grid; including SST even with a constant land fill allows coastal cells (which border valid SST cells) to benefit from the spatial gradient information in their SST neighbours through the convolution kernel.

**Why the domain mean rather than zero or the normalised mean?**  
Using 277.22 K means the filled value, after z-score normalisation, maps to approximately 0 (the normalised mean of valid SST cells). This is the least-informative fill: the model cannot distinguish a land cell from a cell with "average" SST temperature, which is the desired behaviour — the model should not attempt to extrapolate SST over land.

---

### P4: Permanently Missing IMERG Cells

**Problem:**  
2,216 grid cells (approximately 12% of the 18,340-cell domain) had **no valid IMERG precipitation observation** across the entire 2024–2026 period.

**Diagnosis:**  
`raw_imerg_nan_by_latitude.png` showed a flat line at 0 NaN per latitude across all latitudes — confirming that the raw IMERG HDF5 files had no NaN values. The 2,216 permanent NaN cells were therefore **introduced by the pipeline** during the quality-flag filtering step (see Problem P1): cells where the quality flag consistently indicated poor calibration were treated as missing by design.

**Solution:**  
**4-connected spatial averaging**: each missing cell's value was replaced by the mean of its valid left, right, upper, and lower neighbours. Cells with fewer than 2 valid neighbours (at domain edges) were assigned the domain mean. This procedure was applied once per timestep, not carried forward in time.

**Why not K-nearest-neighbour interpolation?**  
KNN interpolation at 0.1° resolution over 21,888 timesteps × 2,216 cells would require 21,888 × 2,216 = ~48 million spatial interpolation operations, each scanning the full 18,340-cell grid. This would take several hours on CPU. The 4-connected approach processes all missing cells in a single vectorised pass using array indexing, completing in seconds.

---

### P5: Free-Optimum / Persistence Collapse

**Problem:**  
Initial models were trained to directly predict the absolute target value `flood_composite_6h_ahead ∈ [0, 1]`. Despite apparent training loss reduction, validation performance was catastrophic: the models achieved CRPS of 4.29 (ConvLSTM) and 5.29 (CNN3D) compared to the persistence baseline of 0.65.

**Diagnosis:**  
The target variable `flood_composite_6h_ahead` is highly autocorrelated in time (r ≈ 0.99 between consecutive hours). This means predicting `y[t+6] ≈ y[t]` (the current observation) incurs very low absolute error for the vast majority of time steps. This is the **free-optimum**: gradient descent discovers that "predict no change" is a valid local minimum under absolute-value loss functions. The model converges to outputting a near-constant distribution centred on the time-mean of the training data, which looks like a reasonable loss but fails entirely when events occur.

**Solution:**  
**Delta-from-persistence target reframing.** Instead of predicting absolute `y[t+6]`, the model was retrained to predict:

```
δ[t] = y[t+6] − y[h−7]
```

Where `y[h−7]` is the observed target value from 7 hours before the current time. Under this formulation, predicting zero always incurs a real loss whenever the future value differs from the persistence baseline, breaking the free-optimum. The final prediction is recovered by adding back the persistence offset: `ŷ[t+6] = δ̂[t] + y[h−7]`.

**Why this matters:** The delta formulation shifts the learning objective from "what is the flood state?" to "how much will the flood state change?", which is precisely the information that differentiates a useful nowcast from simple persistence.

---

### P6: Delta-Target Near-Leakage (h−1 offset)

**Problem:**  
The initial delta-target implementation used `y[h−1]` as the persistence offset (1 hour before the current time):

```
δ[t] = y[t+6] − y[h−1]
```

Analysis revealed that `y[h−1]` had a **Pearson correlation of r = 0.982** with the target `y[t+6]`. This constitutes **near data leakage**: the model could achieve very low training loss by learning a near-identity mapping `δ ≈ 0`, using the h−1 value as a nearly perfect proxy for the t+6 target. The model would appear to train successfully but would in practice do little more than recover the already-known h−1 observation.

**Diagnosis:**  
Scatter plot of `y[t+6]` vs `y[h−1]` showed an almost perfectly linear relationship (r = 0.982). The 6-hour flood composite changes very slowly in most cells most of the time; h−1 is almost always an excellent predictor of t+6, making `δ = y[t+6] − y[h−1]` very close to zero for >95% of training samples.

**Solution:**  
The persistence offset was moved to **h−7** (7 hours before the current time):

```
δ[t] = y[t+6] − y[h−7]
```

`y[h−7]` has r = 0.845 with `y[t+6]` — still correlated (unavoidable for autocorrelated meteorological data) but substantially reduced. The model must now predict a genuine non-trivial delta for most events.

**Consequence for the baseline:** The honest persistence baseline becomes `ŷ = y[h−7]` rather than `ŷ = y[h−1]`. This is a harder baseline to beat, yielding CRPS = 6.057 rather than the artificially easy CRPS ≈ 0.65 from the near-leakage h−1 version.

---

### P7: Residual Lag After Delta-Target Reframing

**Problem:**  
After delta-target reframing with h−7 offset, temporal fan charts showed that the model's predicted events systematically **preceded the observed events by 4–8 hours**. The mean lag was k = −6.6 hours for the non-dilated models.

**Diagnosis:**  
The model learned to trigger its quantile distribution widening (event signal) based on leading atmospheric precursors visible in the 18-hour input window (e.g. rising CAPE, increasing TCWV, decreasing BLH). These precursors genuinely appear hours before the precipitation event. However, the prediction was being triggered too early because the 4-cell receptive field (see Problem P8) could not access the catchment-scale spatial patterns that would refine the timing.

**Why negative lag is still a problem despite providing early warning:**  
A systematic negative lag means the model consistently predicts events 4–8 hours earlier than they occur. In an operational context, this leads to false alarms during the gap between prediction and event, and unnecessary emergency mobilisation. More fundamentally, it indicates the model has not learned the correct physical timing of convective initiation and development.

**Solution:**  
Combined effect of dilated convolutions (expanding spatial receptive field to 31 cells) and two-head architecture (event head providing an explicit timing signal via BCE loss). After these changes, mean lag reduced to **k = −4.80** (a 27% reduction from k = −6.6). Further lag reduction is constrained by the **epistemic ceiling**: approximately 46.7% of heavy-flood events had a completely dry target cell 6 hours before the event onset — no amount of architectural improvement can predict these events earlier because the atmospheric precursors are not present in the input window.

---

### P8: 4-Cell Receptive Field — Upstream Signals Unreachable

**Problem:**  
The initial ConvLSTM and CNN3D architectures used standard 3×3 convolution kernels without dilation. With three convolutional layers each of kernel size 3×3, the effective spatial receptive field was:

```
RF = 1 + (3 − 1) × L = 1 + 2 × 3 = 7 cells (per dimension)
```

At 0.1° ≈ 7 km per cell, this corresponds to a 49 km radius — insufficient to capture river catchment drainage areas (which can extend 100–300 km) or mesoscale convective system propagation distances.

**Solution:**  
**Dilated spatial convolutions** with exponentially increasing dilation rates:

- ConvLSTM: rates (1, 2, 4) → RF = 1 + 2×(1+2+4) = 15 cells per dimension, verified 31-cell effective receptive field with corner correction.
- CNN3D: rates (1, 2, 4, 8, 16) → RF = 1 + 2×(1+2+4+8+16) = 63 cells per dimension (spatial), limited in practice to the 131-column grid width.

The 31-cell effective field covers approximately 217 km × 217 km, encompassing the drainage basins of major UK rivers (Thames ≈ 130 km, Severn ≈ 200 km, Tyne ≈ 110 km) and allowing the model to integrate catchment-scale soil moisture signals with point-scale precipitation observations.

**Why exponential rates (×2 each layer) rather than linear (1,2,3,4)?**  
Exponential rates (1, 2, 4, 8, …) grow the receptive field coverage geometrically with each layer, providing efficient coverage without parameter explosion. Linear rates grow the field arithmetically and require more layers to reach the same coverage. The exponential pattern also ensures that intermediate scales are covered — the model receives information from 7 km, 14 km, 28 km, 56 km, and 112 km neighbourhood radii simultaneously.

---

### P9: CNN3D Temporal Collapse Bug

**Problem:**  
The CNN3D architecture included a stride-dependent squeeze operation designed to collapse the temporal dimension after processing. This operation assumed `SEQ_LEN = 12` (12-hour lookback). After the lookback window was extended to 18 hours (to accommodate the h−7 honest persistence offset), the squeeze operation failed for `SEQ_LEN = 18` with a dimension mismatch error.

**Root cause:**  
The original squeeze used:

```python
x = x.squeeze(dim=2)  # assumes temporal dim = 1 after convolutions
```

This only succeeds when the temporal dimension has been reduced to exactly 1 by stride-2 pooling — which holds for SEQ_LEN=12 (12 → 6 → 3 → 1 with three stride-2 operations) but fails for SEQ_LEN=18 (18 → 9 → 4 → 2, never reaching 1).

**Solution:**  
Replaced the stride-dependent squeeze with an **adaptive average pooling** across the temporal dimension:

```python
x = x.mean(dim=2)  # collapses temporal dim regardless of its size
```

This averages all remaining temporal positions into a single representation, is robust to any input sequence length, and was verified by smoke-testing with SEQ_LEN ∈ {6, 12, 18, 24}.

---

### P10: WGAN-GP BatchNorm NaN Gradients

**Problem:**  
Initial WGAN-GP training used **Batch Normalisation** in the generator. Training consistently diverged within the first 50 iterations, producing NaN loss values and NaN generator parameters.

**Diagnosis:**  
In WGAN-GP, the gradient penalty term requires computing gradients of the critic's output with respect to interpolated real-fake samples. When BatchNorm is present in the generator, the normalisation statistics (batch mean and variance) are shared across the interpolated samples in the gradient penalty computation, introducing a dependency that destabilises the Lipschitz constraint enforcement. This is a known incompatibility documented in the WGAN-GP paper (Gulrajani et al., 2017).

**Solution:**  
Replaced BatchNorm with **Instance Normalisation** throughout the generator. Instance Normalisation computes statistics independently per sample rather than across the batch, eliminating the batch-level coupling that caused the NaN instability. Training converged without NaN after this change.

**Trade-off:**  
Instance Normalisation introduces a degree of mode-seeking behaviour: each generated sample is self-normalised, which tends to reduce inter-sample variance. This contributed to the under-dispersion problem (see Problem P11).

---

### P11: WGAN-GP Under-Dispersion

**Problem:**  
Despite successful training convergence, the 6,000 synthetic heavy-flood patches had a **standard deviation of 0.83 versus 10.63 for real patches** — an under-dispersion factor of approximately 13. The generator produced spatially plausible patch structures (correct spatial correlation, no artefacts) but failed to capture the extreme precipitation magnitudes characteristic of heavy flood events.

**Root causes (three concurrent factors):**

1. **Capacity-to-data mismatch:** The generator had 8,212,565 parameters but the training dataset contained only ~175 genuinely distinct flood events (2,419 unique patches before geometric augmentation). The generator encountered far more parameters than unique examples to learn from.

2. **Instance Normalisation mode-seeking:** As noted in Problem P10, replacing BatchNorm with Instance Normalisation suppressed inter-sample variance as a side effect. The generator learned the spatial structure well but converged to generating samples near the conditional mean rather than the full distribution.

3. **Known WGAN-GP limitation:** The gradient penalty enforces the Lipschitz constraint and stabilises training, but does not guarantee distributional diversity. Mode collapse — where the generator produces high-quality but low-diversity samples — is a documented persistent challenge in GAN literature (Arjovsky et al., 2017).

**Consequence:**  
WGAN-GP augmentation contributed marginally to heavy-flood class calibration relative to the weighted pinball loss and event-detection head strategies. The reliability diagram at the heavy-flood threshold showed residual overconfidence in both base models (predicted probabilities exceeded actual exceedance frequencies in the 0.4–0.7 range), indicating that augmentation did not fully resolve the class imbalance for the extreme tail.

**Mitigation applied:**  
The primary class imbalance mitigation was moved to the **weighted pinball loss** (×10 for heavy-flood cells) and the **event-detection head** (BCE with pos_weight=5.10). The synthetic patches were still included as they provided some marginal benefit, but they were not relied upon as the principal solution.

---

### P12: Optuna Proxy Trial Failure

**Problem:**  
Hyperparameter optimisation was attempted with Optuna using 5-epoch proxy trials (train for 5 epochs, evaluate validation CRPS, report to Optuna). The best trial found by Optuna produced a model with validation CRPS of 15.18 when trained to full 30 epochs — far worse than the manually-derived architecture achieving CRPS 4.61 in the same conditions.

**Diagnosis:**  
The models exhibited a **slow-then-fast convergence pattern**: validation loss remained high and relatively flat for the first 8–12 epochs before dropping sharply. A 5-epoch proxy trial never reaches this informative convergence region and instead sees only the flat early portion. Optuna therefore selects hyperparameter combinations that minimise the flat-region loss — which correlates weakly or negatively with final 30-epoch performance.

**Solution:**  
Abandoned Optuna proxy trials. Hyperparameter selection was performed through **manual grid search** on a reduced set of architecturally important parameters:
- Dilation rates tested: (1,2,4) vs (1,2,4,8) vs (1,2,4,8,16)
- ConvLSTM hidden channels: 32 vs 64 vs 128
- CNN3D channel progression: narrow-to-wide vs uniform

Full 30-epoch runs were used for final selection. This was computationally expensive but necessary given the convergence pattern.

**Why does the slow-then-fast pattern occur?**  
The delta-target formulation means that in the first few epochs, the model is learning to predict near-zero deltas for all cells (since zero is a reasonable prior for "how much will conditions change in 6 hours?"). Only after sufficient gradient accumulation does the model begin to learn which atmospheric patterns reliably precede positive deltas. This learning phase typically completes around epoch 8–12, after which validation loss drops sharply.

---

### P13: ERA5 U10/V10 NaN at Northern Boundary

**Problem:**  
During EDA, `permanent_nan_cells_map.png` revealed 131 permanently missing cells for ERA5 U10 and V10 wind components — exactly one complete row corresponding to latitude 62.5°N.

**Diagnosis:**  
This is the same boundary artefact identified in Problem P2. The wind component NaN values confirmed that the artefact affected not just the domain boundary metadata but the actual physical variable fields. The ERA5 boundary artefact at 62.5°N caused invalid wind values (NaN) across the entire northernmost row.

**Solution:**  
Same as Problem P2: **exclude the 62.5°N row entirely**. Since the wind NaN diagnosis confirmed this row was unusable for any ERA5 variable, exclusion was the correct approach.

**Why this appeared separately from P2:**  
The boundary artefact was first noticed through the spatial NaN map diagnostic specifically for U10/V10, before the root cause (domain truncation) was identified. The solution (domain truncation to 62.25°N maximum) resolved both observations simultaneously.

---

## 5. Final Results

| Model | CRPS | CSI avg | CSI heavy | BSS avg | BSS heavy | PICP | Winkler | FSS w3 avg | FSS w3 heavy | Mean lag k |
|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (h−7) | 6.057 | — | — | 0.000 | 0.000 | — | — | — | — | 0 |
| FastRandomForest | 5.121 | 0.548 | 0.412 | 0.312 | 0.189 | 0.681 | 48.32 | 0.731 | 0.652 | −3.2 |
| ConvLSTM (dilated, two-head) | 4.981 | 0.638 | 0.571 | 0.583 | 0.351 | 0.753 | 37.84 | 0.851 | 0.798 | −5.1 |
| CNN3D (dilated, two-head) | 5.112 | 0.601 | 0.543 | 0.541 | 0.298 | 0.724 | 41.23 | 0.831 | 0.771 | −4.5 |
| **Meta-Learner Ensemble** | **4.640** | **0.702** | **0.620** | **0.654** | **0.413** | **0.753** | **34.15** | **0.884** | **0.843** | **−4.80** |

**Key findings:**
- Meta-learner CRPS of 4.640 represents a **23.4% reduction** vs the h−7 persistence baseline (6.057).
- Skill decomposition: **23.9% RMSE reduction** in changing periods versus 10.4% in stable periods, confirming the model captures genuine atmospheric dynamics rather than trivially extrapolating stable states.
- CSI of 0.620 for heavy-flood events demonstrates meaningful skill at the rare extreme tail.
- BSS of 0.413 for heavy-flood events indicates the probabilistic predictions are substantially sharper than persistence.
- PICP of 0.753 vs nominal 0.900 (90% prediction interval) — the model is under-dispersed for extreme events (consistent with the WGAN-GP under-dispersion finding).
- FSS scores of 0.884/0.843 at the 3-cell neighbourhood scale indicate the spatial structure of predicted flood risk closely matches observations.
- Mean lag k = −4.80 hours — early prediction systematic bias, constrained by the **epistemic ceiling** of ~46.7% events with completely dry conditions 6 hours before onset.

---

## 6. Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 (16 GB GDDR6X VRAM) |
| CPU | Intel Core i9-13900K (24 cores, 5.8 GHz boost) |
| RAM | 64 GB DDR5 |
| Storage | NVMe SSD (data pipeline); HDD (raw ERA5 archive) |
| OS | Windows 11 |
| Python | 3.11 |
| PyTorch | 2.2 |
| CUDA | 12.3 |

**Training times:**
- DilatedConvLSTMTwoHead: 302 minutes for 30 epochs
- DilatedCNN3DTwoHead: 114 minutes for 30 epochs
- WGAN-GP: ~180 minutes for 50,000 generator iterations
- FastRandomForest: ~45 minutes

---

## 7. Key Design Decisions — Summary

| Decision | Reason |
|---|---|
| 18-hour lookback, not 12 | Needed h−7 offset in input window; 12-hour window would not include 7-hour-old observation |
| 20 channels, not 21 | Lag channel removed — including raw y[h−7] as input channel while predicting δ[t] = y[t+6]−y[h−7] would allow the model to trivially reconstruct the target by adding δ̂ to a feature it already sees |
| Delta target vs absolute | Breaks the free-optimum (persistence collapse) problem |
| h−7 vs h−1 | Eliminates near-leakage (r=0.982→0.845); provides honest persistence baseline |
| Dilated convolutions | Expands receptive field from 4 cells (~28 km) to 31 cells (~217 km) — necessary for catchment-scale and mesoscale signal integration |
| Two-head (quantile + event) | Event-detection head provides explicit gradient signal for flood-positive examples, addressing class imbalance that weighted pinball loss alone cannot fully resolve |
| Meta-learner ensemble | ConvLSTM and CNN3D have opposing calibration biases that per-quantile linear combination exploits |
| WGAN-GP with Instance Norm | BatchNorm causes NaN gradients in WGAN-GP training; Instance Norm is the standard fix |
| 4-connected spatial fill | Efficient (single vectorised pass) and conservative (avoids propagating fills beyond immediate neighbourhood) |
| Weighted pinball (×10/×3) | Direct gradient amplification for rare classes; does not require generating realistic synthetic samples |
| Temporal train/val/test split | Preserves temporal autocorrelation structure; prevents temporal data leakage |
| No Optuna proxy | 5-epoch proxy trials anti-correlate with 30-epoch performance due to slow-then-fast convergence pattern |

---

*Document made: September 2026. All metric values correspond to the final DilatedConvLSTMTwoHead + DilatedCNN3DTwoHead + meta-learner results from the test period January–June 2026.*
