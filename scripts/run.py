"""One-shot entry point for the DKW / TIPS inflation-compensation monitor.

    python run.py                      # today, refresh if cache is stale
    python run.py --refresh-bbg        # force a fresh Bloomberg pull
    python run.py --asof 2026-09-16    # reproducible historical run
    python run.py --no-bbg             # Fed-side only (works without Terminal)
    python run.py --outdir <path>      # where the .md brief lands

Prints a one-line signal summary to stdout and writes
`<outdir>/UST_IC_<asof>.md`.
"""
from __future__ import annotations

import argparse
import sys
import traceback
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import attribution, classify, render, risk_z  # noqa: E402
from bbg_fetch import fetch_bbg, skill_root  # noqa: E402
from bridge import run_bridge  # noqa: E402
from dkw_fetch import check_identities, fetch_csv, load_dkw, vintage_notes  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="DKW / TIPS inflation-compensation monitor")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD; default = latest available")
    ap.add_argument("--outdir", default=None, help="report directory")
    ap.add_argument("--refresh-fed", action="store_true", help="force Fed CSV re-download")
    ap.add_argument("--refresh-bbg", action="store_true", help="force full Bloomberg re-pull")
    ap.add_argument("--no-bbg", action="store_true", help="skip Bloomberg entirely")
    ap.add_argument("--window", type=int, default=750, help="rolling estimation window")
    ap.add_argument("--terse", action="store_true", help="print only the headline line")
    return ap.parse_args(argv)


def _wrap(text: str, width: int = 78) -> str:
    """Wrap a CJK paragraph for terminal output (CJK glyphs count double)."""
    def w(ch: str) -> int:
        return 2 if unicodedata.east_asian_width(ch) in "WF" else 1

    no_lead = "，。；、）」》%"
    lines, cur, n = [], "", 0
    for ch in text:
        cw = w(ch)
        if n + cw > width and ch not in no_lead and not cur.endswith(("（", "「")):
            lines.append(cur)
            cur, n = "", 0
        cur += ch
        n += cw
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def main(argv=None) -> int:
    a = parse_args(argv)

    csv = fetch_csv(force=a.refresh_fed)
    dkw = load_dkw(csv)
    ident = check_identities(dkw)
    if not ident["ok"].all():
        print("WARNING: DKW accounting identities failed — schema or source changed",
              file=sys.stderr)

    bbg = None
    bbg_err = ""
    if not a.no_bbg:
        try:
            bbg = fetch_bbg(force=a.refresh_bbg)
        except Exception as e:                                   # noqa: BLE001
            bbg_err = f"{type(e).__name__}: {e}"
            if not a.terse:
                print(f"[warn] Bloomberg unavailable -> Fed-side only ({bbg_err})", file=sys.stderr)

    asof = pd.Timestamp(a.asof) if a.asof else (
        bbg.index.max() if bbg is not None else dkw.index.max())
    if bbg is not None:
        bbg = bbg.loc[bbg.index <= asof]
    dkw = dkw.loc[dkw.index <= asof]

    bridge_res = None
    if bbg is not None:
        try:
            bridge_res = run_bridge(dkw, bbg, asof=asof, window=a.window)
        except Exception as e:                                   # noqa: BLE001
            if not a.terse:
                print(f"[warn] bridge failed: {type(e).__name__}: {e}", file=sys.stderr)
            if "--debug" in sys.argv:
                traceback.print_exc()

    outdir = Path(a.outdir) if a.outdir else (skill_root() / "reports")
    path, plain = render(dkw, bbg, bridge_res, outdir, asof)

    # ---- terminal headline -------------------------------------------------- #
    fed_last = dkw["ic.raw.10"].dropna().index.max()
    fed_ic = dkw["ic.raw.10"].dropna().iloc[-1]
    now_ic = float(bbg["be_10y"].iloc[-1]) if bbg is not None else np.nan
    if np.isfinite(now_ic):
        chg = (now_ic - fed_ic) * 100
        rz = risk_z(bbg)
        attr = attribution(dkw, "10", 21)
        if bridge_res is not None and bridge_res.ok and len(bridge_res.nowcast):
            nc = bridge_res.nowcast
            last = lambda c: dkw[c].dropna().iloc[-1]          # noqa: E731
            attr = pd.Series({
                "ΔIC(市场)": chg,
                "预期通胀": (nc["exp.inflation.10"].iloc[-1] - last("exp.inflation.10")) * 100,
                "通胀风险溢价": (nc["inflation.risk.prem.10"].iloc[-1] - last("inflation.risk.prem.10")) * 100,
                "−TIPS流动性溢价": -(nc["tips.liq.prem.10"].iloc[-1] - last("tips.liq.prem.10")) * 100,
            })
            attr["残差(未解释)"] = attr["ΔIC(市场)"] - (
                attr["预期通胀"] + attr["通胀风险溢价"] + attr["−TIPS流动性溢价"])
        label, _ = classify(attr, rz)
        rr = ((dkw["nominal.yield.raw.10"] - dkw["nominal.yield.fitted.10"])
              - (dkw["ic.raw.10"] - dkw["ic.fitted.10"])).dropna().iloc[-1] * 100
        band = None
        if bridge_res is not None and bridge_res.ok:
            band = bridge_res.metrics.loc[
                bridge_res.metrics["target"] == "ic.fitted.10", "band95_h_bp"]
            band = float(band.iloc[0]) if len(band) else None
        btxt = f" ±{band:.1f}" if band is not None else ""
        same_day = asof <= fed_last
        head = (f"[{asof.date()}] 10y IC 2.36% Fed / {now_ic:.2f}% mkt, "
                f"源间基差 {chg:+.1f}bp（同日，非市场变动）"
                if same_day else
                f"[{asof.date()}] 10y IC {fed_ic:.2f}% -> {now_ic:.2f}% "
                f"({chg:+.1f}bp vs Fed {fed_last.date()})")
        print(
            f"{head} | "
            f"预期 {attr['预期通胀']:+.1f}{btxt} 溢价 {attr['通胀风险溢价'] + attr['−TIPS流动性溢价']:+.1f} "
            f"残差 {attr['残差(未解释)']:+.1f} | {label} | "
            f"TIPS vs 模型 {rr:+.1f}bp | {path.name}"
        )
    else:
        print(f"[{asof.date()}] Fed-only run (no Bloomberg) | 10y IC {fed_ic:.2f}% "
              f"as of {fed_last.date()} | 报告 {path.name}")

    # ---- plain-language read (always printed unless --terse) ---------------- #
    if not a.terse and plain:
        print()
        print("=== 大白话 ===")
        print(_wrap(plain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
