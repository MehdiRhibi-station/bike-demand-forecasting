"""
Six-level hierarchical fallback table for bike demand forecasting.

Each level groups training rows by a progressively coarser key and stores
the *median* demand within that group (medians, not means -- bike demand is
zero-heavy count data, and means get pulled around by a handful of
high-traffic hours). Lookup walks from the most specific level to the
most general, returning the first level with enough support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Ordered from most specific to most general. Each tuple is the set of
# columns that define a group at that level.
FALLBACK_LEVELS: list[tuple[str, ...]] = [
    ("city", "hour", "weekday"),
    ("city", "hour"),
    ("city",),
    ("hour", "weekday"),
    ("hour",),
    (),  # global median -- the last-resort fallback
]

MIN_GROUP_SIZE = 5


@dataclass
class HierarchicalFallback:
    """Median demand lookup across FALLBACK_LEVELS, fit on training data."""

    demand_col: str = "count"
    tables: dict[tuple[str, ...], pd.Series] = field(default_factory=dict)
    global_median: float = 0.0

    def fit(self, df: pd.DataFrame) -> "HierarchicalFallback":
        self.global_median = float(df[self.demand_col].median())
        for level in FALLBACK_LEVELS:
            if not level:
                continue
            grouped = df.groupby(list(level))[self.demand_col]
            medians = grouped.median()
            sizes = grouped.size()
            # Only keep groups with enough support to trust the median.
            self.tables[level] = medians[sizes.reindex(medians.index) >= MIN_GROUP_SIZE]
        return self

    def predict_row(self, row: pd.Series) -> float:
        for level in FALLBACK_LEVELS:
            if not level:
                return self.global_median
            table = self.tables.get(level)
            if table is None:
                continue
            key = tuple(row[col] for col in level) if len(level) > 1 else row[level[0]]
            if key in table.index:
                return float(table.loc[key])
        return self.global_median

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(self.predict_row, axis=1)
