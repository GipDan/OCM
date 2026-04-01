"""Canonical key for a feature_order list (used in DB PK and CSV grouping)."""

from __future__ import annotations

import json


def make_feature_order_key(feature_order: list[str]) -> str:
    """Stable string for the ordered feature name list; same order → same key."""
    return json.dumps(feature_order, ensure_ascii=False, separators=(",", ":"))
