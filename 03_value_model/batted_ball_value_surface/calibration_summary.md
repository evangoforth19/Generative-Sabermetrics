# Batted-ball outcome surrogate — calibration summary

## Data
- **Dataset path:** `/home/evangoforth03/Bayesian Research/data/statcast_pybaseball/statcast_all.parquet`
- **Rows loaded:** 280,398
- **Rows after EV/LA/SA + outcome filters (no bunts / excluded events):** 45,080

## Model
- **Features (only):** `['launch_speed', 'launch_angle', 'sin_spray_angle', 'cos_spray_angle']`
- **Target classes (order):** `['out', 'single', 'double', 'triple', 'home_run']`
- **Split:** stratified 70% / 15% / 15% (fallback: chronological split not viable)
- **Train / validation / test sizes:** 31,556 / 6,762 / 6,762

### Class distribution (counts and %)
**Train:**
```json
{
  "out": {
    "count": 21017,
    "pct": 66.60223095449359
  },
  "single": {
    "count": 6598,
    "pct": 20.90886043858537
  },
  "double": {
    "count": 2067,
    "pct": 6.550259855494993
  },
  "home_run": {
    "count": 1700,
    "pct": 5.387248066928635
  },
  "triple": {
    "count": 174,
    "pct": 0.5514006844974014
  }
}
```
**Validation:**
```json
{
  "out": {
    "count": 4504,
    "pct": 66.60751257024549
  },
  "single": {
    "count": 1414,
    "pct": 20.910973084886127
  },
  "double": {
    "count": 443,
    "pct": 6.551316178645371
  },
  "home_run": {
    "count": 364,
    "pct": 5.383022774327122
  },
  "triple": {
    "count": 37,
    "pct": 0.5471753918958888
  }
}
```
**Test:**
```json
{
  "out": {
    "count": 4503,
    "pct": 66.59272404614019
  },
  "single": {
    "count": 1414,
    "pct": 20.910973084886127
  },
  "double": {
    "count": 443,
    "pct": 6.551316178645371
  },
  "home_run": {
    "count": 365,
    "pct": 5.3978112984324165
  },
  "triple": {
    "count": 37,
    "pct": 0.5471753918958888
  }
}
```

### LightGBM hyperparameters
```json
{
  "n_estimators": 1000,
  "learning_rate": 0.03,
  "num_leaves": 63,
  "max_depth": -1,
  "min_child_samples": 200,
  "subsample": 0.8,
  "colsample_bytree": 0.9,
  "reg_alpha": 0.1,
  "reg_lambda": 1.0,
  "class_weight": "balanced"
}
```
- **Class weighting:** balanced (LGBMClassifier)

## Calibration metrics

| Metric | Validation (before) | Validation (after) | Test (before) | Test (after) |
|--------|----------------------|--------------------|---------------|--------------|
| Multiclass NLL | 0.64825 | 0.62146 | 0.62552 | 0.60493 |
| Multiclass Brier | 0.37370 | 0.36407 | 0.35897 | 0.35151 |
| ECE (top-1, 15 bins) | 0.06340 | 0.01123 | 0.05639 | 0.02013 |

- **Fitted temperature T:** 1.390498

### Class-wise Brier (test, after calibration)
```json
{
  "out": 0.15706272451290298,
  "single": 0.11409814148664156,
  "double": 0.051095252553689956,
  "triple": 0.00901486943263367,
  "home_run": 0.02023763140112324
}
```

### Class-wise one-vs-rest ECE (test, after calibration, 15 bins)
```json
{
  "out": 0.16015587013753546,
  "single": 0.0801301473986596,
  "double": 0.05499762410550728,
  "triple": 0.013996875077835489,
  "home_run": 0.013804295641772683
}
```

### Class-wise reliability (test, after calibration)
Mean predicted probability vs empirical frequency of each class.
```json
{
  "out": {
    "mean_predicted_probability": 0.5057713703238664,
    "empirical_frequency": 0.665927240461402
  },
  "single": {
    "mean_predicted_probability": 0.28914492505011086,
    "empirical_frequency": 0.20910973084886128
  },
  "double": {
    "mean_predicted_probability": 0.12009965621337446,
    "empirical_frequency": 0.06551316178645371
  },
  "triple": {
    "mean_predicted_probability": 0.017332130170633234,
    "empirical_frequency": 0.0054717539189588875
  },
  "home_run": {
    "mean_predicted_probability": 0.06765191824201496,
    "empirical_frequency": 0.053978112984324166
  }
}
```

## Plain-English interpretation
Lower NLL and Brier on validation/test after temperature scaling indicate better probability quality for simulation use. ECE summarizes how much average predicted confidence deviates from realized hit frequencies (smaller is better). Triples are extremely sparse; treat their predicted probabilities as noisy unless counts are large.

> **Warning:** Training set triple count is only **174** (< 500). Triple calibration and probabilities may be unstable for Monte Carlo use.

## Sanity-check predictions (calibrated)

| EV | LA | SA | P(out) | P(single) | P(double) | P(triple) | P(home_run) | xwOBAcon_3D |
|----|----|----|--------|-----------|-----------|-----------|-------------|--------------|
| 105 | 25 | 0 | 0.0457 | 0.0105 | 0.1360 | 0.0672 | 0.7407 | 1.8312 |
| 80 | -10 | 10 | 0.5649 | 0.4304 | 0.0045 | 0.0001 | 0.0001 | 0.3944 |
| 98 | 12 | -25 | 0.1228 | 0.7113 | 0.1649 | 0.0007 | 0.0004 | 0.8542 |
| 110 | 35 | 35 | 0.0049 | 0.0011 | 0.0019 | 0.0030 | 0.9889 | 2.0634 |
