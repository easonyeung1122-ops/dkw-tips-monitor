"""Bloomberg daily market data for the DKW monitor.

All tickers below were validated live against a running Bloomberg session
(bbcomm.exe) — see references/methodology.md. Two gotchas:

  * TIPS generic yields are ZERO-PADDED: `USGGT05YR Index` (5y), not `USGGT5YR`.
  * There is no 15y nominal generic and no direct 5y5y forward ticker.
    15y is therefore skipped; 5y5y forwards are computed from 5y/10y points.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

NOMINAL = {
    "2y": "USGG2YR Index",
    "3y": "USGG3YR Index",
    "5y": "USGG5YR Index",
    "7y": "USGG7YR Index",
    "10y": "USGG10YR Index",
    "20y": "USGG20YR Index",
    "30y": "USGG30YR Index",
}
REAL = {
    "5y": "USGGT05YR Index",
    "10y": "USGGT10YR Index",
    "20y": "USGGT20YR Index",
    "30y": "USGGT30YR Index",
}
BREAKEVEN = {
    "5y": "USGGBE05 Index",
    "10y": "USGGBE10 Index",
    "20y": "USGGBE20 Index",
    "30y": "USGGBE30 Index",
}
RISK = {
    "vix": "VIX Index",
    "hy_oas": "LF98OAS Index",
    "move": "MOVE Index",
    # Brent, NOT WTI: front-month WTI settled at -37.63 on 2020-04-20, which
    # makes a log-return NaN and poisons the regression. Brent stayed positive.
    "oil_brent": "CO1 Comdty",
    "oil_wti": "CL1 Comdty",
    "inflswap_5y": "USSWIT5 Curncy",
    "inflswap_10y": "USSWIT10 Curncy",
}

FIELDS = ("PX_LAST", "YLD_YTM_MID")

# flat column names: "<ticker key>"
ALL_TICKERS: dict[str, str] = {}
for grp, prefix in ((NOMINAL, "nom"), (REAL, "real"), (BREAKEVEN, "be"), (RISK, "risk")):
    for k, v in grp.items():
        ALL_TICKERS[f"{prefix}_{k}"] = v


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cache_path() -> Path:
    return skill_root() / "data" / "bbg_daily.csv"


def _probe_available() -> bool:
    try:
        import blpapi  # noqa: F401
    except Exception:
        return False
    return True


def _pull(tickers: dict[str, str], start: str, end: str, batch: int = 5) -> pd.DataFrame:
    """HistoricalDataRequest in small batches, merging PARTIAL_RESPONSE events."""
    import blpapi

    sess = blpapi.Session()
    if not sess.start():
        raise RuntimeError("blpapi: cannot start session (is bbcomm / Terminal running?)")
    if not sess.openService("//blp/refdata"):
        sess.stop()
        raise RuntimeError("blpapi: cannot open //blp/refdata")
    svc = sess.getService("//blp/refdata")

    series: dict[str, dict[str, float]] = {}
    keys = list(tickers)
    for i in range(0, len(keys), batch):
        chunk = keys[i:i + batch]
        req = svc.createRequest("HistoricalDataRequest")
        for k in chunk:
            req.append("securities", tickers[k])
        req.append("fields", "PX_LAST")
        req.set("startDate", start)
        req.set("endDate", end)
        req.set("periodicitySelection", "DAILY")
        req.set("nonTradingDayFillOption", "NON_TRADING_WEEKDAYS")
        req.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")
        sess.sendRequest(req)
        done = False
        while not done:
            ev = sess.nextEvent(60000)
            et = ev.eventType()
            if et not in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                continue
            for msg in ev:
                if not msg.hasElement("securityData"):
                    continue
                sd = msg.getElement("securityData")
                sec = sd.getElementAsString("security")
                key = {v: k for k, v in tickers.items()}.get(sec, sec)
                bucket = series.setdefault(key, {})
                if sd.hasElement("fieldData"):
                    fda = sd.getElement("fieldData")
                    for j in range(fda.numValues()):
                        fx = fda.getValueAsElement(j)
                        d = fx.getElementAsString("date")
                        if fx.hasElement("PX_LAST") and fx.getElement("PX_LAST").isValid():
                            bucket[d] = fx.getElement("PX_LAST").getValueAsFloat()
            if et == blpapi.Event.RESPONSE:
                done = True
    sess.stop()

    if not series:
        raise RuntimeError("blpapi: no data returned")
    out = pd.DataFrame(series)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def fetch_bbg(start: str = "19970101", end: str | None = None,
              force: bool = False, dest: Path | None = None) -> pd.DataFrame:
    """Cached Bloomberg pull; incrementally extends an existing cache."""
    dest = Path(dest) if dest else cache_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    end = end or dt.date.today().strftime("%Y%m%d")

    cached = None
    if dest.exists() and not force:
        cached = pd.read_csv(dest, index_col=0, parse_dates=True)
        if len(cached) and cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=4):
            return cached
        req_start = (cached.index.max() - pd.Timedelta(days=10)).strftime("%Y%m%d")
    else:
        req_start = start

    fresh = _pull(ALL_TICKERS, req_start, end)
    if cached is not None and len(cached):
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    merged = merged.loc[merged.index >= pd.Timestamp(start)]
    # Bloomberg honours nonTradingDayFillOption per security, so the union index
    # comes back on a full calendar grid. The Fed DKW series is published on ALL
    # weekdays (US holidays included), so normalise to Mon-Fri + short ffill.
    merged = merged[merged.index.dayofweek < 5]
    merged = merged.dropna(how="all")
    merged = merged.ffill(limit=4)
    merged.to_csv(dest)
    return merged


def real_yield(df: pd.DataFrame, tenor: str) -> pd.Series:
    """TIPS real yield. Uses the generic TII ticker; falls back to nom - BE."""
    col = f"real_{tenor}"
    if col in df.columns and df[col].notna().any():
        return df[col]
    return df[f"nom_{tenor}"] - df[f"be_{tenor}"]


def forward(df: pd.DataFrame, short: str, long: str) -> dict[str, pd.Series]:
    """Annualised (long-short) forward from two par-yield points, compounded."""
    n = int(long.rstrip("y")) - int(short.rstrip("y"))
    s = int(short.rstrip("y"))
    fac = (1 + df[f"nom_{long}"] / 100) ** int(long.rstrip("y")) / (1 + df[f"nom_{short}"] / 100) ** s
    fwd_nom = (fac ** (1 / n) - 1) * 100
    facb = (1 + df[f"be_{long}"] / 100) ** int(long.rstrip("y")) / (1 + df[f"be_{short}"] / 100) ** s
    fwd_be = (facb ** (1 / n) - 1) * 100
    facr = (1 + real_yield(df, long) / 100) ** int(long.rstrip("y")) / (1 + real_yield(df, short) / 100) ** s
    fwd_real = (facr ** (1 / n) - 1) * 100
    return {"nom": fwd_nom, "be": fwd_be, "real": fwd_real}


if __name__ == "__main__":
    import sys
    d = fetch_bbg(force="--force" in sys.argv)
    print(f"rows   : {len(d)}")
    print(f"range  : {d.index.min().date()} -> {d.index.max().date()}")
    print(f"columns: {list(d.columns)}")
    print(d.tail(3).to_string())
