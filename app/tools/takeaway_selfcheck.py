# app/tools/takeaway_selfcheck.py
"""
Mini harness (no Streamlit) — anti-regression for Takeaway Specs (v2).

Run:
  python app/tools/takeaway_selfcheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure /app is on sys.path so `data_chat_core` is importable when running as a script
APP_DIR = Path(__file__).resolve().parents[1]  # .../app
sys.path.insert(0, str(APP_DIR))

from data_chat_core.takeaway_specs import TAKEAWAY_SPECS


def _run_one(spec_id: str, anchors: Dict[str, Any]) -> int:
    spec = TAKEAWAY_SPECS[spec_id]
    txt = spec.deterministic_fn(anchors)
    ok, why, diag = spec.validate_fn(txt, anchors)
    if not ok:
        print(f"[FAIL] {spec_id}: {why}\nTXT={txt}\nDIAG={diag}")
        return 1
    # Negative test: too generic should fail validator
    bad = "To jest ogólny opis bez liczb. Rekomenduję działać."
    ok2, why2, _ = spec.validate_fn(bad, anchors)
    if ok2:
        print(f"[FAIL] {spec_id}: validator accepted generic text")
        return 1
    print(f"[OK] {spec_id}: deterministic passes; validator rejects generic.")
    return 0


def main() -> int:
    fails = 0

    fails += _run_one("cs_pareto", {
        "cutoff": 0.80,
        "n_80": 6,
        "share_top1_pct": 28.0,
        "tail_share_pct": 12.0,
        "metric_name": "wartość sprzedaży",
        "total_value": 1250000,
        "top_segments": ["Segment A", "Segment B", "Segment C"],
    })

    fails += _run_one("cs_price_corridor", {
        "p20_price": 19.9,
        "p80_price": 89.9,
        "corridor_share_pct": 81.5,
        "bin_at_p80": "[80, 90)",
        "nan_price_share_pct": 3.2,
        "metric_name": "wartość sprzedaży",
        "total_value": 1250000,
    })


    # Composition Static — scaled
    fails += _run_one("cs_ranking_topn", {
        "metric": "wartość sprzedaży",
        "cat1": "MSZoning",
        "top_segment": "RL",
        "top_value": 219846749,
        "top_share_pct": 83.23,
        "n_segments": 12,
        "hhi": 7123,
    })

    fails += _run_one("cs_waterfall_contrib", {
        "metric": "wartość sprzedaży",
        "cat1": "MSZoning",
        "top_item": "RL",
        "top_item_value": 219846749,
        "top_item_share_pct": 83.23,
        "n_items": 11,
    })

    fails += _run_one("cs_mix_share", {
        "cat1": "Neighborhood",
        "cat2": "BldgType",
        "top_n": 10,
        "dominant_sub": "1Fam",
        "dominant_sub_share_pct": 72.5,
        "max_group": "NAmes",
        "max_group_share_pct": 12.4,
        "n_groups": 10,
        "n_subs": 5,
    })

    fails += _run_one("cs_marimekko", {
        "cat1": "Neighborhood",
        "cat2": "BldgType",
        "top_n": 10,
        "dominant_sub": "1Fam",
        "dominant_sub_share_pct": 72.5,
        "max_group": "NAmes",
        "max_group_share_pct": 12.4,
        "n_groups": 10,
        "n_subs": 5,
    })

    # Distribution — scaled
    fails += _run_one("dist_boxplot_outliers", {"col":"SalePrice","n":1460,"median":163000,"iqr":86000,"outliers_pct":2.5,"lower_fence":0,"upper_fence":340000})
    fails += _run_one("dist_violin_density", {"col":"SalePrice","n":1460,"median":163000,"p95":325000,"skewness":1.9})
    fails += _run_one("dist_rug_points", {"col":"SalePrice","n":1460,"p99":442000,"outliers_pct":2.5})
    fails += _run_one("dist_ecdf_thresholds", {"col":"SalePrice","n":1460,"p95":325000,"p99":442000})
    fails += _run_one("dist_hist_log", {"col":"SalePrice","n":1460,"median":163000,"p99":442000,"skewness":1.9})
    fails += _run_one("dist_kde_density", {"col":"SalePrice","n":1460,"median":163000,"iqr":86000,"skewness":1.9})
    fails += _run_one("dist_hist_narrow_bins", {"col":"SalePrice","n":1460,"min":34900,"max":755000,"median":163000})

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
