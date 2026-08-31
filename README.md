# Bike Demand Forecasting

Machine learning competition project from Course 67577 (Introduction to Machine Learning), Hebrew University of Jerusalem — predicting bike-share demand per city/hour.

## Approach

The final pipeline blends two signals:

1. **A log1p-target Ridge regression** — demand is heavily right-skewed and zero-heavy, so the model is trained on `log1p(demand)` rather than raw counts, which keeps a handful of high-demand hours from dominating the loss.
2. **A six-level hierarchical fallback table** ([`src/fallback_tables.py`](src/fallback_tables.py)) — median demand grouped by progressively coarser keys (e.g. city+hour+weekday → city+hour → city → hour → weekday → global), so a row with little history still gets a sane estimate instead of leaning on a noisy regression fit.

The two are combined with a **history-gated per-row blend weight**: rows backed by more historical observations lean further toward the Ridge prediction, sparse-history rows lean further toward the fallback hierarchy. See [`src/pipeline.py`](src/pipeline.py).

## Key fixes that moved the needle

- **Medians, not means, in the fallback tables.** Bike demand is zero-heavy count data; means get dragged around by outlier hours, medians don't. Switching fixed a chunk of the MAE gap on low-traffic stations.
- **`DEFAULT_BLEND_WEIGHT` bug.** It was set to `0.00`, which silently zeroed out the Ridge model's contribution for every row using the default — the pipeline was quietly running as a pure fallback-table lookup. Fixing the default let the regression actually contribute, and MAE improved both overall and per-city.

## Repo layout

```
src/
  pipeline.py          # end-to-end training/inference pipeline
  fallback_tables.py   # six-level hierarchical median fallback
```

## Running it

```bash
pip install -r requirements.txt
python src/pipeline.py --train data/train.csv --test data/test.csv --out predictions.csv
```

(Bring your own `train.csv`/`test.csv` in the competition's schema — the original competition data isn't redistributed here.)
