# -*- coding: utf-8 -*-
"""Stage3 SAFE FRONTEND MODE.

Central rendering policy to prevent browser overload:

- KPIs / stats: computed on full dataframe (server-side).
- Charts: rendered on a safe sample (client-side) when SAFE mode is effective.

Universal: no dataset-specific assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


DEFAULT_FORCE_ROWS = 100_000
DEFAULT_SAMPLE_ROWS = 5_000
MIN_SAMPLE_ROWS = 500
MAX_SAMPLE_ROWS = 80_000


@dataclass(frozen=True)
class SafeFrontendConfig:
    enabled: bool
    sample_rows: int
    forced: bool
    reason: str
    df_rows: int

    @property
    def effective(self) -> bool:
        return bool(self.enabled or self.forced)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "sample_rows": int(self.sample_rows),
            "forced": bool(self.forced),
            "reason": str(self.reason),
            "df_rows": int(self.df_rows),
            "effective": bool(self.effective),
        }


def compute_safe_frontend_config(
    *,
    df_rows: int,
    enabled: bool,
    sample_rows: int,
    force_rows: int = DEFAULT_FORCE_ROWS,
) -> SafeFrontendConfig:
    forced = bool(int(df_rows) > int(force_rows))
    sr = int(sample_rows or DEFAULT_SAMPLE_ROWS)
    sr = max(MIN_SAMPLE_ROWS, min(MAX_SAMPLE_ROWS, sr))
    reason = f"rows>{force_rows}:forced" if forced else ("enabled" if enabled else "off")
    return SafeFrontendConfig(
        enabled=bool(enabled),
        sample_rows=sr,
        forced=forced,
        reason=reason,
        df_rows=int(df_rows),
    )


def safe_sample_df(
    df: pd.DataFrame,
    cfg: SafeFrontendConfig,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return df suitable for frontend rendering."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if not cfg.effective:
        return df
    n = len(df)
    if n <= cfg.sample_rows:
        return df
    return df.sample(n=cfg.sample_rows, random_state=int(random_state))


def get_cfg_from_ctx(ctx: Optional[Dict[str, Any]], fallback_rows: int) -> SafeFrontendConfig:
    """Read config from ctx (preferred) or return 'off' config."""
    d = (ctx or {}).get("safe_frontend")
    if isinstance(d, dict):
        try:
            return SafeFrontendConfig(
                enabled=bool(d.get("enabled")),
                sample_rows=int(d.get("sample_rows") or DEFAULT_SAMPLE_ROWS),
                forced=bool(d.get("forced")),
                reason=str(d.get("reason") or ""),
                df_rows=int(d.get("df_rows") or fallback_rows),
            )
        except Exception:
            pass
    return compute_safe_frontend_config(
        df_rows=int(fallback_rows),
        enabled=False,
        sample_rows=DEFAULT_SAMPLE_ROWS,
    )
