# Data — UCI Bank Marketing (`bank-full.csv`)

The dataset is **not stored in this repository** (course rule). Download it with:

```bash
python scripts/get_data.py
```

This saves `data/raw/bank-full.csv` and checks that it has 45,211 rows and a `duration` column.

## Source

| Item | Detail |
|---|---|
| Name | Bank Marketing (UCI Machine Learning Repository, dataset id 222) |
| Page | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Download URL | https://archive.ics.uci.edu/static/public/222/bank+marketing.zip |
| File used | `bank+marketing.zip` → `bank.zip` → **`bank-full.csv`** (semicolon-separated) |
| Not used | `bank-additional-full.csv` (a different 41,188-row, 20-feature version) |
| Accessed | 2 November 2026 |
| SHA-256 of `bank-full.csv` | `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d` |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |

## Citation

> S. Moro, P. Cortez and P. Rita, "A data-driven approach to predict the success of bank telemarketing,"
> *Decision Support Systems*, vol. 62, pp. 22–31, 2014.
>
> S. Moro, P. Rita and P. Cortez, "Bank Marketing," UCI Machine Learning Repository, 2014.
> doi:10.24432/C5K306.

## Contents

* 45,211 phone calls from a Portuguese bank's term-deposit campaigns (May 2008 – Nov 2010).
* 16 input features + target `y`. No empty cells.
* **Target mapping:** `y = "yes"` → **1** (subscribed), `"no"` → **0**.
* **Class balance:** 39,922 no (88.30%), 5,289 yes (11.70%).

| Column | Type | Known before the call? |
|---|---|---|
| age, balance, day, campaign, pdays, previous | numeric | yes |
| job, marital, education, default, housing, loan, contact, month, poutcome | categorical | yes |
| **duration** (call length, seconds) | numeric | **no — only after the call ends (leak)** |

## Handling decisions

1. **`"unknown"` is kept as its own category, not imputed.** It appears in `job` (288 rows),
   `education` (1,857), `contact` (13,020) and `poutcome` (36,959). It carries signal:
   `contact = unknown` is almost all in May–June and these clients subscribe at only 4%.
2. **`pdays = -1` means "never contacted before"** (36,954 rows, 81.7%). It is kept as-is
   (tree models can split on it) and a flag `previously_contacted = (pdays != -1)` is added.
3. **`duration` is a target leak.** It is excluded from the deployable feature set. It is only used
   in the FULL feature set to measure how much it inflates scores.
4. **No scaling or log transforms.** All models are tree-based, which only use the order of values.
5. **All encoding happens inside a scikit-learn `Pipeline`**, fitted on training rows only.
   EDA (`notebooks/01_eda.ipynb`) is descriptive and fits nothing.

## Feature sets

| Name | Columns | Purpose |
|---|---|---|
| FULL | 16 original features + `previously_contacted` | reference only: measures the leak |
| DEPLOYABLE | 15 features (no `duration`) + `previously_contacted` | headline results; the model you could ship |

## Split (leak-free, stratified on `y`)

| Part | Share | Rows | Positive rate | Seed |
|---|---|---|---|---|
| Test | 15% | 6,782 | 11.69% | always 42 — **identical for every model and seed** |
| Validation | 15% | 6,782 | 11.69% | 42, 43, 44 |
| Train | 70% | 31,647 | 11.70% | 42, 43, 44 |

* The test set is split off first and is not used until final evaluation.
* For each seed (42, 43, 44) only the train/validation split and the models' `random_state` change.
* Decision-tree depth: 5-fold stratified CV on the training part.
  Forest and boosting hyperparameters and the decision threshold: chosen on validation only.
