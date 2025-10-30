# app/core/config.py
from dataclasses import dataclass
from typing import List, Tuple
import os, platform
from pathlib import Path

REQUIRED = [
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "JWT_SIGNING_KEY",
    "ADMIN_TOKEN",
]

@dataclass
class AppConfig:
    locale: str = "PL"
    openai_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""
    jwt_signing_key: str = ""
    admin_token: str = ""
    enable_tts: bool = True
    max_file_mb: int = 100
    warn_rows: int = 100_000
    sample_rows: int = 500_000
    artifacts_win: str = r"C:\AUTO_EDA_FOR"
    artifacts_nix: str = "~/AUTO_EDA_FOR"

def _mask(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    return (value[:keep] + "…" + "*" * max(0, len(value) - keep - 1)) if len(value) > keep else "*" * len(value)

def load_config() -> Tuple[AppConfig, List[str]]:
    cfg = AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        langfuse_host=os.getenv("LANGFUSE_HOST", ""),
        jwt_signing_key=os.getenv("JWT_SIGNING_KEY", ""),
        admin_token=os.getenv("ADMIN_TOKEN", ""),
        enable_tts=os.getenv("ENABLE_TTS", "true").lower() in ("1","true","yes"),
        max_file_mb=int(os.getenv("MAX_FILE_MB", "100")),
        warn_rows=int(os.getenv("WARN_ROWS", "100000")),
        sample_rows=int(os.getenv("SAMPLE_ROWS", "500000")),
        artifacts_win=os.getenv("ARTIFACTS_WIN", r"C:\AUTO_EDA_FOR"),
        artifacts_nix=os.getenv("ARTIFACTS_NIX", "~/AUTO_EDA_FOR"),
    )
    problems: List[str] = [f"Brak zmiennej: {v}" for v in REQUIRED if not os.getenv(v)]
    return cfg, problems

def snapshot_masked_env() -> dict:
    snap = {var: _mask(os.getenv(var, "")) for var in REQUIRED}
    snap["MAX_FILE_MB"] = os.getenv("MAX_FILE_MB","100")
    snap["WARN_ROWS"] = os.getenv("WARN_ROWS","100000")
    snap["SAMPLE_ROWS"] = os.getenv("SAMPLE_ROWS","500000")
    snap["ENABLE_TTS"] = os.getenv("ENABLE_TTS","true")
    snap["ARTIFACTS_WIN"] = os.getenv("ARTIFACTS_WIN", r"C:\AUTO_EDA_FOR")
    snap["ARTIFACTS_NIX"] = os.getenv("ARTIFACTS_NIX", "~/AUTO_EDA_FOR")
    return snap

def resolve_artifacts_dir(cfg: AppConfig) -> Path:
    """Zwraca istniejący katalog artefaktów zgodny z OS."""
    base = Path(cfg.artifacts_win) if platform.system() == "Windows" else Path(os.path.expanduser(cfg.artifacts_nix))
    base.mkdir(parents=True, exist_ok=True)
    return base
