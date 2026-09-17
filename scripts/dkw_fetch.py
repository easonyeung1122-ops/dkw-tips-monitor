"""Fed DKW decomposition CSV: download, cache, parse.

Source
------
D'Amico, Kim & Wei (2018), JFQA 53(1):395-436, updated by
Kim, Walsh & Wei (2019), FEDS Notes, "Tips from TIPS: Update and Discussions".

The published CSV is NOT the 2019 methodology. Check `vintage_notes()` output:
the current vintage re-estimates the model on a fixed sample and re-applies it
monthly, and has switched to BCFF monthly 1-year forecasts plus 15y/20y yields
in the estimation. Always report the vintage you actually used.
"""
from __future__ import annotations

import datetime as dt
import io
import urllib.request
from pathlib import Path

import pandas as pd

CSV_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/DKW_updates.csv"
NOTE_URL = (
    "https://www.federalreserve.gov/econres/notes/feds-notes/"
    "tips-from-tips-update-and-discussions-20190521.html"
)
UA = "Mozilla/5.0 (compatible; dkw-tips-monitor/1.0)"

TENORS = ("5", "10", "5f5")
TENOR_LABEL = {"5": "5y", "10": "10y", "5f5": "5y5y fwd"}
COMPONENTS = (
    "exp.real.short.rate",
    "exp.inflation",
    "real.term.prem",
    "inflation.risk.prem",
    "tips.liq.prem",
)
DERIVED = ("nominal.yield.raw", "nominal.yield.fitted", "ic.raw", "ic.fitted")

# published column order per tenor block
BLOCK_COLS = (
    "exp.real.short.rate",
    "exp.inflation",
    "real.term.prem",
    "inflation.risk.prem",
    "tips.liq.prem",
    "nominal.yield.raw",
    "nominal.yield.fitted",
    "ic.raw",
    "ic.fitted",
)


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cache_path() -> Path:
    return skill_root() / "data" / "DKW_updates.csv"


def fetch_csv(force: bool = False, max_age_days: int = 7, dest: Path | None = None) -> Path:
    """Download the Fed CSV if missing or stale. Returns the local path."""
    dest = Path(dest) if dest else cache_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        age = (dt.datetime.now() - dt.datetime.fromtimestamp(dest.stat().st_mtime)).days
        if age < max_age_days:
            return dest

    req = urllib.request.Request(CSV_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    if len(raw) < 10_000:
        raise RuntimeError(f"Fed CSV looks truncated ({len(raw)} bytes)")
    dest.write_bytes(raw)
    return dest


def _split(path: Path) -> tuple[list[str], int]:
    """Return (lines, index of the column-header line)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        first = ln.replace('"', "").split(",")[0].strip().lower()
        if first == "date":
            return lines, i
    raise RuntimeError("could not locate column header row in DKW CSV")


def vintage_notes(path: Path | None = None) -> list[str]:
    """The prose note lines embedded above the header (methodology vintage)."""
    path = Path(path) if path else (cache_path() if cache_path().exists() else fetch_csv())
    lines, idx = _split(path)
    out = []
    for ln in lines[:idx]:
        s = ln.strip().strip('"').strip()
        if s and s.lower() not in {"notes:", "source:"}:
            out.append(s)
    return out


def load_dkw(path: Path | None = None, force: bool = False) -> pd.DataFrame:
    """Parsed DKW series, DatetimeIndex ascending, numeric columns.

    All rates are percentage points. `tips.liq.prem.*` and `ic.*` are NaN
    before TIPS were issued (1997); the pre-1997 rows are still useful for
    the nominal-rate decomposition.
    """
    if path is None:
        path = fetch_csv(force=force)
    path = Path(path)

    lines, idx = _split(path)
    body = "\n".join(lines[idx:])
    df = pd.read_csv(io.StringIO(body), na_values=["NA", "na", "", " "])
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("date").drop_duplicates("date").set_index("date")

    missing = [f"{c}.{t}" for t in TENORS for c in BLOCK_COLS if f"{c}.{t}" not in df.columns]
    if missing:
        raise RuntimeError(f"DKW CSV schema changed, missing columns: {missing[:6]} ...")
    return df


def check_identities(df: pd.DataFrame, tol: float = 2e-4) -> pd.DataFrame:
    """Verify the two accounting identities on the published data.

    nominal.yield.fitted = exp.real.short.rate + exp.inflation + real.term.prem
                           + inflation.risk.prem
    ic.fitted            = exp.inflation + inflation.risk.prem - tips.liq.prem
    """
    rows = []
    for t in TENORS:
        n_fit = (
            df[f"exp.real.short.rate.{t}"] + df[f"exp.inflation.{t}"]
            + df[f"real.term.prem.{t}"] + df[f"inflation.risk.prem.{t}"]
        )
        ic_fit = (
            df[f"exp.inflation.{t}"] + df[f"inflation.risk.prem.{t}"]
            - df[f"tips.liq.prem.{t}"]
        )
        rows.append({
            "tenor": TENOR_LABEL[t],
            "nominal_identity_max_abs_err": float((n_fit - df[f"nominal.yield.fitted.{t}"]).abs().max()),
            "ic_identity_max_abs_err": float((ic_fit - df[f"ic.fitted.{t}"]).abs().max()),
            "ok": bool(
                (n_fit - df[f"nominal.yield.fitted.{t}"]).abs().max() < tol
                and (ic_fit - df[f"ic.fitted.{t}"]).abs().max() < tol
            ),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = fetch_csv(force="--force" in __import__("sys").argv)
    d = load_dkw(p)
    print(f"file      : {p}")
    print(f"rows      : {len(d)}")
    print(f"range     : {d.index.min().date()} -> {d.index.max().date()}")
    print(f"columns   : {d.shape[1]}")
    print("\nvintage notes:")
    for n in vintage_notes(p):
        print("  -", n)
    print("\nidentity check:")
    print(check_identities(d).to_string(index=False))
