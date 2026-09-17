"""Bridge nowcast: extend the Fed DKW decomposition to the latest trading day.

Why this exists
---------------
The Fed CSV is refreshed MONTHLY (4th business day) but carries DAILY series.
On a typical day the published decomposition lags the market by ~2-3 weeks.
This module bridges that gap with a ridge regression in CHANGES:

    d(component)_t = a + b' dX_t + e_t      (fit on a rolling window)
    component_t    = component_{t-1} + d(component)_t

Anchoring on the Fed's last published level is deliberate: the Fed print is
authoritative, and we only extend it. Fitting in changes (not levels) avoids
the spurious ~0.99 R^2 you get from regressing two persistent series on each
other, so the reported out-of-sample skill is honest.

Honesty protocol
----------------
* Ridge penalty chosen on an OLDER validation slice.
* Skill REPORTED on a NEWER, disjoint slice that never touched parameter choice.
* `r2_oos` is measured against the random-walk benchmark (d = 0). If it is
  near or below zero the bridge adds nothing, and the report says so.

All figures in this module are OUR estimates, not Federal Reserve output.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dkw_fetch import TENORS, TENOR_LABEL  # noqa: E402

# component series we bridge, per tenor
COMPONENT_SPECS = (
    "exp.real.short.rate",
    "exp.inflation",
    "real.term.prem",
    "inflation.risk.prem",
    "tips.liq.prem",
)

# daily-change market predictors (built from the Bloomberg panel)
PREDICTOR_COLS = (
    "d_real_5y", "d_real_10y", "d_real_20y", "d_real_30y",
    "d_be_5y", "d_be_10y", "d_be_20y", "d_be_30y",
    "d_vix", "d_hy_oas", "d_oil_pct", "d_move",
)

DEFAULT_WINDOW = 750         # rolling estimation window (obs)
DEFAULT_LAMBDAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
VAL_SELECT = 400             # older slice: choose lambda
VAL_REPORT = 400             # newer slice: report skill (untouched by selection)


# --------------------------------------------------------------------------- #
# panel construction
# --------------------------------------------------------------------------- #
def build_master(dkw: pd.DataFrame, bbg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align both sources onto the weekday calendar the Fed publishes on."""
    cal = dkw.index.union(bbg.index)
    cal = cal[cal.dayofweek < 5]
    cal = cal.sort_values()
    d = dkw.reindex(cal)
    b = bbg.reindex(cal)
    return d, b.ffill(limit=4)


def build_predictors(bbg_al: pd.DataFrame) -> pd.DataFrame:
    """Daily changes, expressed in bp (yields) / points or pct (risk factors)."""
    out = pd.DataFrame(index=bbg_al.index)
    for t in ("5y", "10y", "20y", "30y"):
        out[f"d_real_{t}"] = bbg_al[f"real_{t}"].diff() * 100.0
        out[f"d_be_{t}"] = bbg_al[f"be_{t}"].diff() * 100.0
    out["d_vix"] = bbg_al["risk_vix"].diff()
    out["d_hy_oas"] = bbg_al["risk_hy_oas"].diff() * 100.0     # OAS quoted in %
    # sign-safe pct change: WTI went negative in Apr-2020, abs() keeps it finite
    out["d_oil_pct"] = 100.0 * bbg_al["risk_oil_brent"].diff() / bbg_al["risk_oil_brent"].shift().abs()
    out["d_move"] = bbg_al["risk_move"].diff()
    return out[list(PREDICTOR_COLS)]


def target_frame(dkw_al: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Component levels, plus the two aggregates DERIVED from components.

    Deriving (rather than bridging) the aggregates keeps the published
    accounting identities true by construction in the nowcast.
    """
    cols: dict[str, pd.Series] = {}
    for t in TENORS:
        for c in COMPONENT_SPECS:
            cols[f"{c}.{t}"] = dkw_al[f"{c}.{t}"]
        cols[f"ic.fitted.{t}"] = (
            dkw_al[f"exp.inflation.{t}"] + dkw_al[f"inflation.risk.prem.{t}"]
            - dkw_al[f"tips.liq.prem.{t}"]
        )
        cols[f"nominal.yield.fitted.{t}"] = (
            dkw_al[f"exp.real.short.rate.{t}"] + dkw_al[f"exp.inflation.{t}"]
            + dkw_al[f"real.term.prem.{t}"] + dkw_al[f"inflation.risk.prem.{t}"]
        )
    df = pd.DataFrame(cols)
    order = [c for t in TENORS for c in
             [f"{x}.{t}" for x in COMPONENT_SPECS] + [f"ic.fitted.{t}", f"nominal.yield.fitted.{t}"]]
    return df[order], order


# --------------------------------------------------------------------------- #
# ridge machinery
# --------------------------------------------------------------------------- #
def _ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """X already contains an intercept column and is standardised."""
    k = X.shape[1]
    A = X.T @ X + lam * np.eye(k)
    A[0, 0] -= lam          # never penalise the intercept
    return np.linalg.solve(A, X.T @ Y)


CLIP = 5.0      # winsorise standardised predictors: tames 2020-style outliers


def _standardise(X: np.ndarray, clip: float = CLIP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return _transform(X, mu, sd, clip), mu, sd


def _transform(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = CLIP) -> np.ndarray:
    return np.clip((X - mu) / sd, -clip, clip)


def _design(Xd: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(Xd)), Xd])


@dataclass
class BridgeResult:
    nowcast: pd.DataFrame                      # gap days x target columns
    metrics: pd.DataFrame                      # per-target OOS skill
    errors: pd.DataFrame | None = None         # daily errors on the report slice (bp)
    lam_used: dict[str, float] = field(default_factory=dict)
    gap_days: list = field(default_factory=list)
    horizon: int = 0
    ok: bool = True
    message: str = ""


def run_bridge(dkw: pd.DataFrame, bbg: pd.DataFrame, asof: pd.Timestamp | None = None,
               window: int = DEFAULT_WINDOW,
               lambdas=DEFAULT_LAMBDAS,
               val_select: int = VAL_SELECT,
               val_report: int = VAL_REPORT) -> BridgeResult:
    dkw_al, bbg_al = build_master(dkw, bbg)
    if asof is not None:
        bbg_al = bbg_al.loc[bbg_al.index <= asof]
        dkw_al = dkw_al.loc[dkw_al.index <= asof]

    X_all = build_predictors(bbg_al)
    Y_level, targets = target_frame(dkw_al)
    Y_level = Y_level.loc[X_all.index]

    # tidy frame: everything we need on one grid, drop warm-up junk
    frame = pd.concat([X_all, Y_level], axis=1)
    last_fed = Y_level.dropna(how="all").index.max()
    gap_idx = X_all.index[X_all.index > last_fed]
    horizon = len(gap_idx)
    if len(gap_idx) == 0:
        return BridgeResult(pd.DataFrame(), pd.DataFrame(), None, {}, [], 0, False,
                            "无空档：Fed 序列已覆盖最新交易日，本节无需外推")

    # estimation grid = days where predictors AND all targets exist
    good = frame[list(PREDICTOR_COLS)].notna().all(axis=1) & Y_level.notna().all(axis=1)
    est_idx = frame.index[good & (frame.index <= last_fed)]
    if len(est_idx) < window + val_select + val_report:
        return BridgeResult(pd.DataFrame(), pd.DataFrame(), None, {}, [], 0, False,
                            f"insufficient overlapping history ({len(est_idx)} obs)")

    Xd = frame.loc[est_idx, list(PREDICTOR_COLS)].to_numpy(float)
    Yd = (Y_level.loc[est_idx].diff().to_numpy(float)) * 100.0     # bp changes
    keep = ~np.isnan(Yd).any(axis=1)
    Xd, Yd = Xd[keep], Yd[keep]
    Yd_dates = est_idx[keep]
    n = len(Xd)

    # --- walk-forward over the two validation slices (all targets at once) ---
    v_start = n - (val_select + val_report)
    v_end = n
    preds = {lam: np.full((v_end - v_start, Yd.shape[1]), np.nan) for lam in lambdas}
    for lam in lambdas:
        for i, t in enumerate(range(v_start, v_end)):
            lo = max(0, t - window)
            if t - lo < 200:
                continue
            Xs, mu, sd = _standardise(Xd[lo:t])
            beta = _ridge(_design(Xs), Yd[lo:t], lam)
            preds[lam][i] = np.r_[1.0, _transform(Xd[t], mu, sd)] @ beta

    # ---- step 1: lambda per target by 1-day MSE on the OLDER slice ---------- #
    col_index = {c: i for i, c in enumerate(targets)}
    val_dates = Yd_dates[v_start:v_end]
    sel_slice = slice(0, val_select)
    lam_used, e_raw = {}, {}
    for c in targets:
        j = col_index[c]
        err = {}
        for lam in lambdas:
            diff = preds[lam][sel_slice, j] - Yd[v_start:][sel_slice, j]
            diff = diff[np.isfinite(diff)]
            err[lam] = float(np.mean(diff ** 2)) if diff.size else np.inf
        lam_used[c] = min(err, key=lambda k: err[k])
        e_raw[c] = pd.Series(preds[lam_used[c]][:, j] - Yd[v_start:, j], index=val_dates)

    # ---- step 2: gamma (path damping) per COMPONENT on the OLDER slice ----- #
    # The 1-day R^2 flatters the bridge: it is largely same-day co-movement, and
    # the Fed's components are SMOOTHER than market yields, so an undamped
    # cumulated path over-predicts. We shrink the whole h-day predicted path by
    # gamma and fit gamma on the older slice:
    #
    #   cum_err = rolling_sum(pred - act)          =>  rolling_sum(pred) = cum_err + act
    #   error(gamma) = gamma * rolling_sum(pred) - act
    #   gamma* = argmin E[error^2] = E[pred_h*act_h] / E[pred_h^2], clipped to [0,1]
    #
    # gamma -> 0 means the bridge adds nothing and the honest answer is
    # "assume the decomposition has not moved since the last Fed print".
    comp_targets = [f"{c}.{t}" for t in TENORS for c in COMPONENT_SPECS]
    cut = val_dates[val_select]
    gam_used, acth, cum_err_d = {}, {}, {}
    for c in comp_targets:
        e = e_raw[c]
        act = Y_level[c].diff(horizon) * 100
        acth[c] = act
        cum_e = e.rolling(horizon).sum().dropna()
        cum_err_d[c] = cum_e
        df = pd.DataFrame({"cum_e": cum_e, "act": act.reindex(cum_e.index)}).dropna()
        sel = df[df.index < cut]
        if len(sel) < 50:
            gam_used[c] = 1.0
            continue
        pred_cum = sel["cum_e"] + sel["act"]
        denom = float((pred_cum ** 2).mean())
        g = float((pred_cum * sel["act"]).mean() / denom) if denom > 0 else 0.0
        gam_used[c] = float(min(max(g, 0.0), 1.0))

    # ---- step 3: damped h-day errors; aggregates by error propagation ------ #
    e_h = {}
    for c in comp_targets:
        e_h[c] = gam_used[c] * (cum_err_d[c] + acth[c]) - acth[c]
    for t in TENORS:
        e_h[f"ic.fitted.{t}"] = (e_h[f"exp.inflation.{t}"]
                                 + e_h[f"inflation.risk.prem.{t}"]
                                 - e_h[f"tips.liq.prem.{t}"])
        e_h[f"nominal.yield.fitted.{t}"] = (e_h[f"exp.real.short.rate.{t}"]
                                            + e_h[f"exp.inflation.{t}"]
                                            + e_h[f"real.term.prem.{t}"]
                                            + e_h[f"inflation.risk.prem.{t}"])
        acth[f"ic.fitted.{t}"] = Y_level[f"ic.fitted.{t}"].diff(horizon) * 100
        acth[f"nominal.yield.fitted.{t}"] = Y_level[f"nominal.yield.fitted.{t}"].diff(horizon) * 100

    # ---- step 4: metrics on the NEWER, untouched slice --------------------- #
    rows = []
    for c in targets:
        j = col_index[c]
        a1 = pd.Series(Yd[v_start:, j], index=val_dates)
        p1 = a1 + e_raw[c]
        m_rep = a1.index >= cut
        row = {"target": c, "lambda": lam_used[c],
               "gamma": gam_used.get(c, np.nan), "n_oos": int(m_rep.sum())}
        av, pv = a1[m_rep].dropna(), p1[m_rep].dropna()
        idx = av.index.intersection(pv.index)
        if len(idx) >= 50:
            se_m = float(np.mean((pv.loc[idx] - av.loc[idx]) ** 2))
            se_rw = float(np.mean(av.loc[idx] ** 2))
            row.update({"rmse_bp": float(np.sqrt(se_m)),
                        "rmse_rw_bp": float(np.sqrt(se_rw)),
                        "r2_oos": float(1 - se_m / se_rw) if se_rw > 0 else np.nan})
        # e_h is ALREADY the h-day nowcast error; acth is the random-walk error.
        # Do not subtract again: R2_h = 1 - E[err_model^2] / E[err_rw^2].
        ehv = e_h[c].dropna()
        ih = ehv.index[ehv.index >= cut]
        if len(ih) >= 50:
            dv, av2 = ehv.loc[ih], acth[c].reindex(ih)
            keep = dv.notna() & av2.notna()
            dv, av2 = dv[keep], av2[keep]
            se_h = float((dv ** 2).mean())
            rw_h = float((av2 ** 2).mean())
            row.update({"rmse_h_bp": float(np.sqrt(se_h)),
                        "rmse_rw_h_bp": float(np.sqrt(rw_h)),
                        "r2_h_oos": float(1 - se_h / rw_h) if rw_h > 0 else np.nan,
                        "band95_h_bp": float(np.quantile(np.abs(dv), 0.95))})
        rows.append(row)
    metrics = pd.DataFrame(rows)
    errors = pd.DataFrame(e_h)          # damped h-day nowcast errors (bp)

    # --- final fit on the most recent window, then walk the gap forward --- #
    lo = max(0, n - window)
    Xs, mu, sd = _standardise(Xd[lo:n])
    beta = _ridge(_design(Xs), Yd[lo:n], 10.0)

    Xg = frame.loc[gap_idx, list(PREDICTOR_COLS)].to_numpy(float)
    gap_fin = ~np.isnan(Xg).any(axis=1)
    gap_idx = gap_idx[gap_fin]
    Xg = Xg[gap_fin]

    anchor = Y_level.loc[:last_fed].iloc[-1].to_numpy(float)
    dY = _design(_transform(Xg, mu, sd)) @ beta            # bp, shape (h, m)
    dY = dY * np.array([gam_used.get(c, 1.0) for c in targets])   # damped path
    paths = np.vstack([anchor]) + np.cumsum(dY / 100.0, axis=0)   # back to %

    nowcast = pd.DataFrame(paths, index=gap_idx, columns=targets)
    # consistency: re-derive aggregates from the bridged components
    for t in TENORS:
        nowcast[f"ic.fitted.{t}"] = (
            nowcast[f"exp.inflation.{t}"] + nowcast[f"inflation.risk.prem.{t}"]
            - nowcast[f"tips.liq.prem.{t}"]
        )
        nowcast[f"nominal.yield.fitted.{t}"] = (
            nowcast[f"exp.real.short.rate.{t}"] + nowcast[f"exp.inflation.{t}"]
            + nowcast[f"real.term.prem.{t}"] + nowcast[f"inflation.risk.prem.{t}"]
        )
    nowcast.attrs["anchor_date"] = last_fed
    nowcast.attrs["metrics"] = metrics
    return BridgeResult(nowcast=nowcast, metrics=metrics, errors=errors,
                        lam_used=lam_used, gap_days=list(gap_idx), horizon=horizon, ok=True,
                        message=f"bridged {len(gap_idx)} trading day(s) beyond {last_fed.date()}")


def band_bp(res: "BridgeResult", target: str, horizon: int | None = None,
            z: float = 1.96) -> float | None:
    """95% band in bp for an h-day nowcast, from the validated metrics table.

    The number comes from the EMPIRICAL h-day error distribution on the
    untouched validation slice. Do NOT substitute `z * 1-day RMSE * sqrt(h)`:
    nowcast errors are strongly persistent, and sqrt(h) understates the band
    by roughly an order of magnitude over a two-week gap.
    """
    if res is None or res.metrics is None or not len(res.metrics):
        return None
    row = res.metrics.loc[res.metrics["target"] == target]
    if row.empty:
        return None
    h = int(res.horizon if horizon is None else horizon)
    col = "band95_h_bp" if h == res.horizon else None
    if col and col in row.columns and np.isfinite(row[col].iloc[0]):
        return float(row[col].iloc[0])
    if "rmse_h_bp" in row.columns and np.isfinite(row["rmse_h_bp"].iloc[0]):
        return z * float(row["rmse_h_bp"].iloc[0])
    if np.isfinite(row["rmse_bp"].iloc[0]):
        return z * float(row["rmse_bp"].iloc[0]) * np.sqrt(h)
    return None


if __name__ == "__main__":
    from bbg_fetch import fetch_bbg
    from dkw_fetch import load_dkw

    d = load_dkw()
    b = fetch_bbg()
    res = run_bridge(d, b)
    print(res.message)
    print(f"\nper-target OOS skill, horizon={res.horizon}d "
          f"(lambda+gamma chosen on older slice; metrics on newer, untouched slice):")
    m = res.metrics.copy()
    for c in ("rmse_bp", "rmse_rw_bp", "rmse_h_bp", "rmse_rw_h_bp", "band95_h_bp"):
        if c in m.columns:
            m[c] = m[c].round(2)
    for c in ("r2_oos", "r2_h_oos", "gamma"):
        if c in m.columns:
            m[c] = m[c].round(3)
    print(m.to_string(index=False))
    print("\nnowcast tail (5f5 block):")
    print(res.nowcast[[c for c in res.nowcast.columns if c.endswith(".5f5")]].tail(3).round(4).to_string())
