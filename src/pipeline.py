"""
End-to-end bike-demand-forecasting pipeline: a log1p-target Ridge
regression blended with the hierarchical fallback in fallback_tables.py.

Usage:
    python src/pipeline.py --train data/train.csv --test data/test.csv --out predictions.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fallback_tables import HierarchicalFallback

NUMERIC_FEATURES = ["temp", "humidity", "windspeed"]
CATEGORICAL_FEATURES = ["city", "hour", "weekday", "season", "weather"]
TARGET = "count"

# Below this many historical observations for a row's (city, hour, weekday)
# group, lean on the fallback table instead of the regression.
HISTORY_GATE_THRESHOLD = 20

# Fixed after debugging a silent bug: this used to default to 0.00, which
# zeroed out the Ridge model's contribution for every row that fell back to
# the default weight. MAE improved overall and per-city once this was fixed.
DEFAULT_BLEND_WEIGHT = 0.65


def build_ridge_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def history_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["city", "hour", "weekday"])["city"].transform("size")


def blend_weight(history: pd.Series) -> pd.Series:
    """More history -> lean toward Ridge. Little history -> lean toward the fallback table."""
    gated = (history >= HISTORY_GATE_THRESHOLD).astype(float)
    return gated * DEFAULT_BLEND_WEIGHT + (1 - gated) * (DEFAULT_BLEND_WEIGHT * 0.3)


def train(train_df: pd.DataFrame) -> tuple[Pipeline, HierarchicalFallback]:
    ridge_pipeline = build_ridge_pipeline()
    ridge_pipeline.fit(
        train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        np.log1p(train_df[TARGET]),
    )

    fallback = HierarchicalFallback(demand_col=TARGET).fit(train_df)
    return ridge_pipeline, fallback


def predict(
    ridge_pipeline: Pipeline,
    fallback: HierarchicalFallback,
    df: pd.DataFrame,
) -> pd.Series:
    ridge_pred = np.expm1(
        ridge_pipeline.predict(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    )
    fallback_pred = fallback.predict(df)

    history = history_counts(df)
    weight = blend_weight(history)

    blended = weight * ridge_pred + (1 - weight) * fallback_pred
    return blended.clip(lower=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    ridge_pipeline, fallback = train(train_df)
    predictions = predict(ridge_pipeline, fallback, test_df)

    test_df.assign(count=predictions)[["count"]].to_csv(args.out, index=False)
    print(f"Wrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
