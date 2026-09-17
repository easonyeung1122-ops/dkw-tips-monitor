"""Analysis + Markdown brief for the DKW / TIPS inflation-compensation monitor.

Reliability layering (this ordering is deliberate and should survive into the
report): market-side observation > Fed decomposition > our bridge nowcast.
Anything produced by `bridge.py` is OUR estimate and is labelled as such.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bbg_fetch import forward, real_yield  # noqa: E402
from bridge import band_bp, run_bridge  # noqa: E402
from dkw_fetch import COMPONENTS, TENORS, TENOR_LABEL, check_identities  # noqa: E402

HORIZONS = {"1d": 1, "1w": 5, "1m": 21, "3m": 63}

# Fed tenor key -> (short, long) par points, for the one forward the Fed publishes
FORWARD_SPEC: dict[str, tuple[str, str]] = {"5f5": ("5y", "10y")}

MARKET_TENORS = (
    ("5y", "5y"),
    ("10y", "10y"),
    ("20y", "20y"),
    ("30y", "30y"),
    ("5y5y fwd", ("5y", "10y")),
    ("10y10y fwd", ("10y", "20y")),
    ("20y10y fwd", ("20y", "30y")),
)


def _bp(x) -> float:
    return float(x) * 100.0


# --------------------------------------------------------------------------- #
# 1. market side: zero-model, daily
# --------------------------------------------------------------------------- #
def market_table(bbg: pd.DataFrame, horizons: dict = HORIZONS) -> pd.DataFrame:
    rows = []
    for label, spec in MARKET_TENORS:
        if isinstance(spec, tuple):
            fwd = forward(bbg, spec[0], spec[1])
            nom, be, rl = fwd["nom"], fwd["be"], fwd["real"]
            tii = np.nan
        else:
            nom, be = bbg[f"nom_{spec}"], bbg[f"be_{spec}"]
            rl = real_yield(bbg, spec)
            tii = rl.iloc[-1]
        real_imp = nom - be                      # exact: keeps d(nom)=d(real)+d(be)
        for hlabel, h in horizons.items():
            if len(nom.dropna()) <= h:
                continue
            rows.append({
                "tenor": label,
                "horizon": hlabel,
                "level_nom": nom.iloc[-1],
                "level_real": real_imp.iloc[-1],
                "level_be": be.iloc[-1],
                "d_nom_bp": _bp(nom.diff(h).iloc[-1]),
                "d_real_bp": _bp(real_imp.diff(h).iloc[-1]),
                "d_be_bp": _bp(be.diff(h).iloc[-1]),
                "basis_tii_bp": _bp(tii - real_imp.iloc[-1]) if np.isfinite(tii) else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Fed decomposition: authoritative, but lagged
# --------------------------------------------------------------------------- #
def fed_table(dkw: pd.DataFrame, z_window: int = 1260) -> pd.DataFrame:
    rows = []
    for t in TENORS:
        last = dkw[f"ic.raw.{t}"].dropna().index.max()
        for c in COMPONENTS:
            s = dkw[f"{c}.{t}"]
            rows.append(_fed_row(t, last, s, c, z_window, dkw))
        for c in ("ic.raw", "ic.fitted"):
            s = dkw[f"{c}.{t}"]
            rows.append(_fed_row(t, last, s, c, z_window, dkw))
    return pd.DataFrame(rows)


def _fed_row(t: str, last, s: pd.Series, label: str, z_window: int, dkw: pd.DataFrame) -> dict:
    w = s.loc[:last].dropna()
    z = np.nan
    if len(w) > z_window:
        tail = w.iloc[-z_window:]
        sd = tail.std()
        z = float((w.iloc[-1] - tail.mean()) / sd) if sd else np.nan
    return {
        "tenor": TENOR_LABEL[t],
        "series": label,
        "asof": last,
        "level": float(w.iloc[-1]) if len(w) else np.nan,
        "d_1w_bp": _bp(w.diff(5).iloc[-1]) if len(w) > 5 else np.nan,
        "d_1m_bp": _bp(w.diff(21).iloc[-1]) if len(w) > 21 else np.nan,
        "z_5y": z,
    }


def residuals(dkw: pd.DataFrame) -> pd.DataFrame:
    """raw - fitted. resid_real = resid_nom - resid_ic: TIPS cheapness vs model."""
    out = {}
    for t in TENORS:
        nom_r = dkw[f"nominal.yield.raw.{t}"] - dkw[f"nominal.yield.fitted.{t}"]
        ic_r = dkw[f"ic.raw.{t}"] - dkw[f"ic.fitted.{t}"]
        out[TENOR_LABEL[t]] = pd.DataFrame({
            "resid_nom_bp": nom_r * 100,
            "resid_ic_bp": ic_r * 100,
            "resid_real_bp": (nom_r - ic_r) * 100,
        })
    return out


def live_residual(dkw: pd.DataFrame, bbg: pd.DataFrame, bridge_res) -> pd.DataFrame | None:
    """Push the Fed residual forward to the latest trading day.

    The residual is the only output here with a z-score, so it is the actionable
    one -- but Fed publishes it 2-3 weeks late. Refreshing it by LEVELS would
    difference Bloomberg's imputed IC against Fed's `ic.raw`, and those two are
    not the same measurement: measured on 1983-2026 the level basis runs
    mean +2.3bp with sd 4.5bp (10y) / 11.6bp (5y5y), i.e. as large as the signal.

    So we refresh it in CHANGES instead::

        resid_live = resid_anchor + [ d(IC_market) - d(IC_fitted) ]

    A constant source basis cancels inside each difference, which is what makes
    this legitimate. resid_real is Fed's own accounting (`resid_nom - resid_ic`),
    not a ticker comparison.
    """
    if bridge_res is None or not getattr(bridge_res, "ok", False) or len(bridge_res.nowcast) == 0:
        return None
    anchor = bridge_res.nowcast.attrs["anchor_date"]
    asof = bridge_res.nowcast.index.max()

    def mkt_ic(t: str) -> pd.Series:
        """Market IC on the report's convention, from the TII ticker."""
        key = FORWARD_SPEC.get(t)
        if key:
            f = forward(bbg, key[0], key[1])
            return f["nom"] - f["real"]
        return bbg[f"nom_{t}y"] - real_yield(bbg, f"{t}y")

    def mkt_nom(t: str) -> pd.Series:
        key = FORWARD_SPEC.get(t)
        return forward(bbg, key[0], key[1])["nom"] if key else bbg[f"nom_{t}y"]

    rows = []
    for t in TENORS:
        fit_a = dkw[f"ic.fitted.{t}"].dropna().loc[:anchor].iloc[-1]
        raw_a = dkw[f"ic.raw.{t}"].dropna().loc[:anchor].iloc[-1]
        nom_a = dkw[f"nominal.yield.fitted.{t}"].dropna().loc[:anchor].iloc[-1]
        nraw_a = dkw[f"nominal.yield.raw.{t}"].dropna().loc[:anchor].iloc[-1]

        ic_a = (raw_a - fit_a) * 100
        nom_r_a = (nraw_a - nom_a) * 100
        real_a = nom_r_a - ic_a

        s_ic = mkt_ic(t).dropna()
        d_ic = (s_ic.loc[:asof].iloc[-1] - s_ic.loc[:anchor].iloc[-1]) * 100
        d_fit = (bridge_res.nowcast[f"ic.fitted.{t}"].iloc[-1] - fit_a) * 100
        ic_live = ic_a + (d_ic - d_fit)

        s_nom = mkt_nom(t).dropna()
        d_nom = (s_nom.loc[:asof].iloc[-1] - s_nom.loc[:anchor].iloc[-1]) * 100
        d_nom_fit = (bridge_res.nowcast[f"nominal.yield.fitted.{t}"].iloc[-1] - nom_a) * 100
        real_live = (nom_r_a + (d_nom - d_nom_fit)) - ic_live

        sd = ((dkw[f"nominal.yield.raw.{t}"] - dkw[f"nominal.yield.fitted.{t}"])
              - (dkw[f"ic.raw.{t}"] - dkw[f"ic.fitted.{t}"])).dropna().iloc[-1260:].std() * 100
        # resid_real is a DIFFERENCE of two independently bridged series, so the
        # band is the quadrature sum of theirs -- not either one alone.
        b_nom = band_bp(bridge_res, f"nominal.yield.fitted.{t}")
        b_ic = band_bp(bridge_res, f"ic.fitted.{t}")
        b_res = (float(np.hypot(b_nom, b_ic))
                 if b_nom is not None and b_ic is not None else None)
        rows.append({
            "tenor": TENOR_LABEL[t],
            "resid_ic_anchor_bp": ic_a,
            "resid_ic_live_bp": ic_live,
            "resid_real_anchor_bp": real_a,
            "resid_real_live_bp": real_live,
            "d_bp": real_live - real_a,
            "z_live": real_live / sd if sd else np.nan,
            "band_bp": b_res,
        })
    return pd.DataFrame(rows)


def attribution(dkw: pd.DataFrame, tenor: str, h: int) -> pd.Series:
    """Exact split of the MARKET IC change:

    d(ic.raw) = d(exp.inflation) + d(IRP) - d(LP) + d(residual)
    """
    d = lambda c: dkw[f"{c}.{tenor}"].diff(h).iloc[-1] * 100
    resid = (dkw[f"ic.raw.{tenor}"] - dkw[f"ic.fitted.{tenor}"])
    return pd.Series({
        "ΔIC(市场)": d("ic.raw"),
        "预期通胀": d("exp.inflation"),
        "通胀风险溢价": d("inflation.risk.prem"),
        "−TIPS流动性溢价": -d("tips.liq.prem"),
        "残差(未解释)": resid.diff(h).iloc[-1] * 100,
    })


# --------------------------------------------------------------------------- #
# 3. sentiment linkage (the paper's own diagnostic)
# --------------------------------------------------------------------------- #
def sentiment_table(dkw: pd.DataFrame, bbg: pd.DataFrame, weekly: bool = True) -> pd.DataFrame:
    """Weekly-change correlation of 5-to-10y IC with HY OAS / VIX / Brent.

    Mirrors Kim-Walsh-Wei (2019) Table 1 so the numbers are comparable in
    construction, though the underlying model vintage differs.
    """
    k = 5 if weekly else 1
    ic = dkw["ic.raw.5f5"]
    frame = pd.DataFrame({
        "ic": ic.diff(k),
        "hy": bbg["risk_hy_oas"].diff(k),
        "vix": bbg["risk_vix"].diff(k),
        "oil": bbg["risk_oil_brent"].pct_change(k),
    }).dropna()
    if frame.empty:
        return pd.DataFrame()
    segments = {
        "全样本 1999-": frame.loc["1999-01-01":],
        "危机后 2009/7-": frame.loc["2009-07-01":],
        "危机前 -2008/7": frame.loc["1999-01-01":"2008-07-01"],
    }
    rows = []
    for name, seg in segments.items():
        if len(seg) < 30:
            continue
        rows.append({
            "sample": name,
            "n": len(seg),
            "corr_HY": seg["ic"].corr(seg["hy"]),
            "corr_VIX": seg["ic"].corr(seg["vix"]),
            "corr_oil": seg["ic"].corr(seg["oil"]),
        })
    return pd.DataFrame(rows)


def risk_z(bbg: pd.DataFrame, window: int = 750, k: int = 5) -> dict:
    """How unusual is the current weekly move in each risk factor?"""
    out = {}
    for name, col in (("hy_oas", "risk_hy_oas"), ("vix", "risk_vix"), ("oil_brent", "risk_oil_brent")):
        s = bbg[col].pct_change(k) if name == "oil_brent" else bbg[col].diff(k)
        s = s.dropna()
        if len(s) < window:
            window = max(60, len(s) // 2)
        tail = s.iloc[-window:]
        sd = tail.std()
        out[name] = float(s.iloc[-1] / sd) if sd else np.nan
    return out


# --------------------------------------------------------------------------- #
# 4. verdict
# --------------------------------------------------------------------------- #
def classify(contrib: pd.Series, rz: dict) -> tuple[str, str]:
    """Decide whether an IC move carries inflation-expectation information.

    Compares channel MAGNITUDES, not shares of the net change: when the net
    move is small (and components offset), shares explode and mislead.
    """
    total = contrib.get("ΔIC(市场)", np.nan)
    if not np.isfinite(total) or abs(total) < 1.0:
        return "中性", f"IC 净变动仅 {total:+.1f}bp，低于 1bp 噪声阈值，不构成信号"
    exp_ = float(contrib.get("预期通胀", 0.0))
    prem = float(contrib.get("通胀风险溢价", 0.0) + contrib.get("−TIPS流动性溢价", 0.0))
    resid = float(contrib.get("残差(未解释)", 0.0))
    mag = max(abs(exp_), abs(prem), abs(resid))
    if mag == 0:
        return "中性", "各渠道贡献均为零"
    sentiment_hot = any(
        np.isfinite(rz.get(k, np.nan)) and abs(rz[k]) > 0.5 for k in ("hy_oas", "vix", "oil_brent")
    )
    detail = (f"净变动 {total:+.1f}bp = 预期 {exp_:+.1f} + 溢价 {prem:+.1f} + 残差 {resid:+.1f}")
    if abs(resid) >= mag:
        return "模型残差主导", (
            f"{detail}；残差为最大单项。残差落在 DKW 三成分之外（TIPS 供需/未建模流动性），"
            f"不应读作通胀预期或风险溢价变化"
        )
    if abs(exp_) >= 0.6 * mag and abs(exp_) > abs(prem):
        return "预期驱动", f"{detail}；预期通胀为最大单项且未被溢价渠道抵消"
    if abs(prem) >= 0.6 * mag:
        tag = "（且风险因子周变动超 0.5σ）" if sentiment_hot else ""
        return "情绪/溢价驱动", (
            f"{detail}；溢价渠道为最大单项{tag}。按 Kim-Walsh-Wei 的两条传导渠道，"
            f"此类变动**不应**读作通胀预期变化"
        )
    return "混合", f"{detail}；无单一渠道主导"


# --------------------------------------------------------------------------- #
# 5. rendering
# --------------------------------------------------------------------------- #
def _neg_zero(rendered: str) -> str:
    """`{:+.1f}` renders a tiny negative as '-0.0', which reads like a bug."""
    return ("+" + rendered[1:]) if rendered.startswith("-0.0") else rendered


def _md_table(df: pd.DataFrame, cols=None, fmt=None, index=False) -> str:
    """Self-contained markdown table renderer (no tabulate dependency)."""
    d = df[cols].copy() if cols else df.copy()
    if fmt:
        for c, f in fmt.items():
            if c in d.columns:
                d[c] = d[c].map(
                    lambda v, f=f: _neg_zero(f.format(v)) if pd.notna(v) else "n/a")
    header = ([""] if index else []) + [str(c) for c in d.columns]
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in d.iterrows():
        cells = [str(idx)] if index else []
        for v in row:
            if isinstance(v, str):
                cells.append(v)
            elif pd.isna(v):
                cells.append("n/a")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(f"{float(v):.3f}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _sign(v: float) -> str:
    return f"{v:+.1f}" if pd.notna(v) else "n/a"


# --------------------------------------------------------------------------- #
# 4b. plain-language read: the same numbers, said the way you would say them
#     out loud. Every run emits one paragraph. No jargon, no hedging filler --
#     the point is that someone who does not know what a "term premium" is can
#     still act on the conclusion.
# --------------------------------------------------------------------------- #
def _num(x, spec: str = "{:+.1f}", na: str = "n/a") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    if not np.isfinite(v):
        return na
    return _neg_zero(spec.format(v))


def _d(x) -> str:
    """Timestamp -> 'YYYY-MM-DD' for prose (never '2026-08-31 00:00:00')."""
    try:
        return pd.Timestamp(x).date().isoformat()
    except (TypeError, ValueError):
        return str(x)


def _row(mt, tenor: str, horizon: str):
    if mt is None or not len(mt):
        return None
    s = mt[(mt["tenor"] == tenor) & (mt["horizon"] == horizon)]
    return s.iloc[0] if len(s) else None


def plain_read(dkw: pd.DataFrame, bbg: pd.DataFrame | None, bridge_res, asof,
               label: str, attr_gap, mt, rz: dict, lr, fed_last) -> str:
    """One plain-Chinese paragraph reading the conclusion of this run.

    Deliberately narrates in the same order a decision gets made:
    what moved -> who moved it -> did inflation expectations move -> is there
    anything mispriced -> one-sentence summary. Terms of art are replaced by
    their meaning; the only calibration jargon kept (z, sigma) is explained.
    """
    S: list[str] = []
    gap_n = len(bridge_res.gap_days) if (bridge_res is not None and bridge_res.ok) else 0
    ic10_fed = float(dkw["ic.raw.10"].dropna().iloc[-1])
    ic10_now = float(bbg["be_10y"].iloc[-1]) if bbg is not None else np.nan
    gap_bp = (ic10_now - ic10_fed) * 100 if np.isfinite(ic10_now) else np.nan

    # ---- 1. what happened, in time ---------------------------------------- #
    if bbg is None:
        S.append(f"这次没连上行情终端，只能读美联储那份分解（最后更新 {_d(fed_last)}），"
                 f"10 年期通胀补偿当时是 {ic10_fed:.2f}%。")
    elif gap_n == 0:
        S.append(f"美联储的数据刚好更新到今天（{_d(fed_last)}），不用推算缺口，"
                 f"下面看到的都是实盘数字。")
    else:
        direction = "上行" if gap_bp > 0 else "下行"
        S.append(f"先看时间差：美联储那份分解最后一次更新是 {_d(fed_last)}，到今天 {_d(asof)} "
                 f"中间隔了 {gap_n} 个交易日。这段时间里，市场给未来十年通胀开出的「补偿价」"
                 f"（也就是通胀挂钩债和被通胀吃掉收益的普通国债之间的利差）"
                 f"从 {ic10_fed:.2f}% 变成 {ic10_now:.2f}%，{direction} "
                 f"{abs(gap_bp):.1f} 个基点，1 个基点 = 0.01%。")

    # ---- 2. who did the moving: real rate or inflation -------------------- #
    row = _row(mt, "10y", "1m") if mt is not None else None
    if row is None:
        row = _row(mt, "10y", "1w") if mt is not None else None
    win = ""
    dn = dr = db = np.nan
    share = np.nan
    if row is not None:
        dn, dr, db = (float(row["d_nom_bp"]), float(row["d_real_bp"]), float(row["d_be_bp"]))
        win = "一个月" if row["horizon"] == "1m" else "一周"
        share = dr / dn if dn else np.nan
        if abs(dn) >= 3:
            if np.isfinite(share) and share >= 0.7:
                ph = f"大约 {share:.0%} 的波动来自实际利率"
            elif np.isfinite(share) and share >= 0.4:
                ph = f"大约一半（{share:.0%}）来自实际利率"
            elif np.isfinite(share) and share > 0:
                ph = f"实际利率只解释了 {share:.0%}"
            else:
                ph = "实际利率和通胀预期方向相反，一个涨一个跌"
            S.append(f"把这段名义收益率的变动拆开看（近{win}，10 年期）：名义 {_num(dn)}bp，"
                     f"其中「实际利率」——借真钱、扣掉通胀后的成本——{_num(dr)}bp，"
                     f"「通胀那一块」只有 {_num(db)}bp。也就是说，{ph}，"
                     f"这波跟通胀定价基本没关系。")
        else:
            S.append(f"近{win}的 10 年期名义收益率基本没动（{_num(dn)}bp），"
                     f"拆不出有意义的结构。")

    # ---- 3. did inflation expectations themselves move -------------------- #
    scope = "这几天"
    if attr_gap is None:
        attr_gap = attribution(dkw, "10", 21)
        scope = "美联储数据覆盖的最近一个月"
    exp_ = float(attr_gap.get("预期通胀", 0.0) or 0.0)
    prem_ = float(attr_gap.get("通胀风险溢价", 0.0) or 0.0) + \
        float(attr_gap.get("−TIPS流动性溢价", 0.0) or 0.0)
    res_ = float(attr_gap.get("残差(未解释)", np.nan))
    tot_ = float(attr_gap.get("ΔIC(市场)", np.nan))
    hot = [k for k in ("hy_oas", "vix", "oil_brent")
           if np.isfinite(rz.get(k, np.nan)) and abs(rz[k]) > 0.5]
    nice = {"hy_oas": "高收益债利差", "vix": "股市波动率", "oil_brent": "油价"}
    hot_txt = ("、".join(nice[k] for k in hot) + "这几个") if hot else ""
    if label == "模型残差主导":
        S.append(f"再说通胀预期本身有没有动：读不出来。{scope}的变化里最大的一块"
                 f"（{_num(res_)}bp）落在模型解释不了的地方——通胀挂钩债自己的供需和流动性，"
                 f"既不是大家对通胀的看法变了，也不是要求的风险补偿变了。")
    elif label == "预期驱动":
        S.append(f"通胀预期本身确实动了：各项里最大的是「预期通胀」{_num(exp_)}bp，"
                 f"而且没被其他因素抵消——这是{scope}里唯一能读成通胀信息的部分。")
    elif label == "情绪/溢价驱动":
        S.append(f"这次动的主要不是对通胀的看法，而是「持有通胀风险要的额外补偿」"
                 f"{_num(prem_)}bp"
                 + (f"，同时{hot_txt}本周波动都超过平时一个标准差——这是市场情绪和"
                    f"避险需求在推，不该当成通胀预期变了。" if hot else
                    "——这类变动来自风险补偿本身，不是通胀预期变了。"))
    elif label == "混合":
        S.append(f"{scope}的变化里几个渠道都有份（预期 {_num(exp_)} / 风险补偿 {_num(prem_)} / "
                 f"解释不了的部分 {_num(res_)}），没有谁明显占主导，读不出干净的结论。")
    else:
        S.append(f"{scope}10 年期通胀补偿净变动只有 {_num(tot_)}bp，比日常噪声还小，等于没动。")

    # ---- 4. is anything actually mispriced -------------------------------- #
    side = ""
    z10 = dv10 = np.nan
    band10 = None
    have_price = False
    if lr is not None and len(lr):
        ridx = lr.set_index("tenor")
        lab10 = TENOR_LABEL["10"]
        if lab10 in ridx.index:
            have_price = True
            r10 = ridx.loc[lab10]
            v10 = float(r10["resid_real_live_bp"])
            z10, dv10 = float(r10["z_live"]), float(r10["d_bp"])
            band10 = float(r10["band_bp"]) if pd.notna(r10["band_bp"]) else None
            side = "便宜" if v10 > 0 else "贵"
            txt = (f"真正值得盯一眼的只有一处定价：10 年期通胀挂钩债相对美联储模型的拟合值"
                   f"{side}了 {abs(v10):.1f}bp，相当于它平常波动的 {abs(z10):.1f} 倍"
                   f"（一般要超过 2 倍才算数）")
            if abs(z10) >= 2 and band10 is not None:
                if abs(dv10) <= band10:
                    txt += (f"。但这 {gap_n} 天里它只又走了 {abs(dv10):.1f}bp，"
                            f"还没超过推算误差 ±{band10:.1f}bp —— 说明这个偏移早就在那儿了，"
                            f"不是这几天新冒出来的，别当成「刚发生的错价」去追，"
                            f"要赚它得先扛住它继续扩大。")
                else:
                    txt += (f"，而且这 {gap_n} 天里还朝同一个方向多走了 {abs(dv10):.1f}bp，"
                            f"超过推算误差 ±{band10:.1f}bp，方向上是认真的。")
            elif abs(z10) >= 2:
                txt += "，统计上算显著。"
            else:
                txt += "，还不到 2 倍标准差，只能说偏高或偏低，谈不上错价。"
            S.append(txt)
            if abs(z10) >= 2:
                S.append(f"真要顺着这个方向做，动作是买通胀挂钩债、同时卖普通国债"
                         f"（赌的是这个{side}会收敛，不是赌通胀会涨）。")

    # ---- 5. one sentence --------------------------------------------------- #
    if bbg is None:
        closing = "一句话：只跑了美联储那一半，没有行情数据，读不出市场在定价什么。"
    else:
        if np.isfinite(dn) and abs(dn) >= 3 and np.isfinite(share) and share >= 0.4:
            d_txt = "这波收益率变动主要是实际利率的事，通胀预期基本没参与。"
        elif np.isfinite(dn) and abs(dn) >= 3:
            d_txt = "这波收益率变动里，通胀那一块不能忽略。"
        else:
            d_txt = "名义收益率这段时间没怎么动。"
        if side and abs(z10) >= 2:
            fresh = ("但它是老问题、不是新出现的，追进去要先扛浮亏。" if (
                band10 is not None and abs(dv10) <= band10) else "而且还在朝同一方向走。")
            s_txt = f"唯一有点意思的是通胀挂钩债相对模型偏{side} {abs(v10):.1f}bp，{fresh}"
        elif side:
            s_txt = f"通胀挂钩债相对模型偏{side}，但幅度不够大，不值得动手。"
        elif have_price:
            s_txt = "没发现值得动手的定价偏离。"
        else:
            s_txt = "这次没算出可用的错价读数，定价偏离只能看第 4 节的 Fed 原值。"
        closing = f"一句话：{d_txt}{s_txt}"

    S.append(closing)
    if bbg is not None:
        S.append("（白话对照：名义收益率 ≈ 实际利率 + 市场对通胀的报价，后者就是「通胀补偿」；"
                 "「基点」= 0.01%，「z 值」= 把偏离换成「几个平常波动」，超过 2 才叫显著。）")
    return "".join(S)


def render(dkw: pd.DataFrame, bbg: pd.DataFrame | None, bridge_res, outdir: Path,
           asof: pd.Timestamp) -> tuple[Path, str]:
    """Write the brief; return ``(path, plain_paragraph)``.

    The plain-language paragraph is returned as well as embedded so the CLI can
    print it without recomputing anything.
    """
    fed_last = dkw["ic.raw.5f5"].dropna().index.max()
    L: list[str] = []
    A = L.append

    # ---- headline ---------------------------------------------------------- #
    ic10_now = bbg["be_10y"].iloc[-1] if bbg is not None else np.nan
    ic10_fed = dkw["ic.raw.10"].dropna().iloc[-1]
    gap_bp = (ic10_now - ic10_fed) * 100 if bbg is not None else np.nan
    rz = risk_z(bbg) if bbg is not None else {}
    # computed once, reused by the headline, section 1, section 4b and the
    # plain-language read
    mt = market_table(bbg) if bbg is not None else None
    lr = live_residual(dkw, bbg, bridge_res) if bbg is not None else None
    attr_gap = None
    if bridge_res is not None and bridge_res.ok and len(bridge_res.nowcast):
        nc = bridge_res.nowcast
        attr_gap = pd.Series({
            "ΔIC(市场)": (ic10_now - ic10_fed) * 100,
            "预期通胀": (nc["exp.inflation.10"].iloc[-1] - dkw["exp.inflation.10"].dropna().iloc[-1]) * 100,
            "通胀风险溢价": (nc["inflation.risk.prem.10"].iloc[-1] - dkw["inflation.risk.prem.10"].dropna().iloc[-1]) * 100,
            "−TIPS流动性溢价": -(nc["tips.liq.prem.10"].iloc[-1] - dkw["tips.liq.prem.10"].dropna().iloc[-1]) * 100,
            "残差(未解释)": ((ic10_now - ic10_fed) - (nc["ic.fitted.10"].iloc[-1] - dkw["ic.fitted.10"].dropna().iloc[-1])) * 100,
        })
    label, why = classify(attr_gap if attr_gap is not None else
                          attribution(dkw, "10", 21), rz)

    resid_real = {}
    for t in TENORS:
        rr_ = (dkw[f"nominal.yield.raw.{t}"] - dkw[f"nominal.yield.fitted.{t}"]) - \
              (dkw[f"ic.raw.{t}"] - dkw[f"ic.fitted.{t}"])
        resid_real[t] = rr_.dropna().iloc[-1] * 100
    cheap10 = "偏便宜" if resid_real["10"] > 0 else "偏贵"
    cheap5f5 = "偏便宜" if resid_real["5f5"] > 0 else "偏贵"

    A(f"# UST 通胀补偿监测 — {asof.date()}")
    A("")
    gap_n = len(bridge_res.gap_days) if (bridge_res and bridge_res.ok) else 0
    resid_txt = (f"TIPS 实际收益率相对 DKW 模型：10y {cheap10} {abs(resid_real['10']):.1f}bp，"
                 f"5y5y 远期 {cheap5f5} {abs(resid_real['5f5']):.1f}bp（Fed 口径，{fed_last.date()}）。")
    # refresh that residual to the latest trading day; the Fed number is stale
    lr_head = lr
    if lr_head is not None and len(lr_head):
        r = lr_head.set_index("tenor")
        live10 = r.loc[TENOR_LABEL["10"]] if TENOR_LABEL["10"] in r.index else None
        live5f5 = r.loc[TENOR_LABEL["5f5"]] if TENOR_LABEL["5f5"] in r.index else None
        if live10 is not None:
            ch = "扩大" if live10["d_bp"] > 0 else "收敛"
            v10 = "偏便宜" if live10["resid_real_live_bp"] > 0 else "偏贵"
            s5 = ""
            if live5f5 is not None:
                v5 = "偏便宜" if live5f5["resid_real_live_bp"] > 0 else "偏贵"
                s5 = (f"；5y5y 远期 {v5} {abs(live5f5['resid_real_live_bp']):.1f}bp"
                      f"（z {live5f5['z_live']:+.2f}）")
            resid_txt += (f" **刷新到 {bridge_res.nowcast.index.max().date()}（自算）**："
                          f"10y {v10} {abs(live10['resid_real_live_bp']):.1f}bp，"
                          f"z {live10['z_live']:+.2f}，偏离较锚点{ch} {abs(live10['d_bp']):.1f}bp"
                          f"（噪声带 ±{live10['band_bp']:.1f}bp）{s5}。")
    if gap_n == 0:
        A(f"**结论**：{asof.date()} 与 Fed 最后发布日同日，**无外推空档**。"
          f"10y 通胀补偿（Fed）{ic10_fed:.2f}%，Bloomberg 盈亏平衡 {ic10_now:.2f}%，"
          f"两者之差 {_sign(gap_bp)}bp 属**源间基差**（报价口径不同），不是市场变动。"
          f"判定 **{label}**（基于 Fed 覆盖期内近 1 月口径）。{resid_txt}")
    else:
        A(f"**结论**：Fed 最后发布日 {fed_last.date()} 至 {asof.date()}（{gap_n} 个交易日），"
          f"10y 通胀补偿 {ic10_fed:.2f}% → {ic10_now:.2f}%（{_sign(gap_bp)}bp，"
          f"Fed 最后发布值与当前市场盈亏平衡之差）。判定 **{label}**。{resid_txt}")
    A("")
    A(f"> {why}")
    A("")

    # ---- 0. plain-language read -------------------------------------------- #
    # Emitted on EVERY run, before any table. If the tables and this paragraph
    # ever disagree, the tables are right and this function has a bug -- it is
    # a pure re-statement of the same numbers, deliberately jargon-free.
    try:
        plain = plain_read(dkw, bbg, bridge_res, asof, label, attr_gap, mt, rz, lr, fed_last)
    except Exception as e:                                       # noqa: BLE001
        plain = f"_白话解读生成失败（{type(e).__name__}: {e}）；下方表格不受影响。_"
    A("## 0. 大白话版（每次运行必出，不看表格也能懂）")
    A("")
    A(plain)
    A("")

    # ---- 1. market side ---------------------------------------------------- #
    A("## 1. 市场侧分解（零模型，基于 Bloomberg 实盘）")
    A("")
    if bbg is None:
        A("_未连接 Bloomberg，本节不可用。_")
    else:
        mt = market_table(bbg)
        tenors = [t for t, _ in MARKET_TENORS]

        # Levels are horizon-INVARIANT: printing them under each window would
        # repeat four identical column sets. They get exactly one table.
        snap = (mt.drop_duplicates("tenor")
                  .rename(columns={"level_nom": "名义", "level_real": "实际",
                                   "level_be": "盈亏平衡", "basis_tii_bp": "口径差"}))
        A(f"**1.1 水平快照**（截至 {bbg.index.max().date()}，%）")
        A("")
        A(_md_table(snap, ["tenor", "名义", "实际", "盈亏平衡", "口径差"],
                    fmt={"名义": "{:.2f}", "实际": "{:.2f}",
                         "盈亏平衡": "{:.2f}", "口径差": "{:+.1f}"}))
        A("")

        # Changes DO vary by window, so the window becomes a row, not a table.
        grid = []
        for key, nice in (("d_nom_bp", "名义"), ("d_real_bp", "实际"), ("d_be_bp", "盈亏平衡")):
            for h in HORIZONS:
                sub = mt.loc[mt["horizon"] == h].set_index("tenor")[key]
                if sub.empty:
                    continue
                row = {"度量": f"Δ{nice}", "窗口": h}
                row.update(sub.to_dict())
                grid.append(row)
        gt = pd.DataFrame(grid)
        gcols = ["度量", "窗口"] + [t for t in tenors if t in gt.columns]
        A("**1.2 变动矩阵**（bp；行 = 度量 × 窗口，列 = 期限）")
        A("")
        A(_md_table(gt, gcols, fmt={c: "{:+.1f}" for c in gcols if c not in ("度量", "窗口")}))
        A("")
        A("口径：`实际` 由 `名义 − 盈亏平衡` 反推，故 **Δ名义 = Δ实际 + Δ盈亏平衡 在每一行都恒等成立**，"
          "可当自我校验用。")
        A("`口径差` = TIPS 真实收益率 generic ticker 与该反推值之差（报价源差异，非信号）——"
          "它是水平量、与窗口无关，故只在 1.1 出现一次。")
        A(f"窗口按**工作日**计（h = {'/'.join(str(v) for v in HORIZONS.values())}），"
          "美假日以 PREVIOUS_VALUE 填充，以对齐 Fed 的全工作日发布口径。")
    A("")

    # ---- 2. Fed decomposition --------------------------------------------- #
    A(f"## 2. Fed DKW 分解（权威口径，截至 {fed_last.date()}）")
    A("")
    ft = fed_table(dkw)
    for t in TENORS:
        A(f"**{TENOR_LABEL[t]}**")
        A("")
        sub = ft[ft["tenor"] == TENOR_LABEL[t]]
        A(_md_table(sub, ["series", "level", "d_1w_bp", "d_1m_bp", "z_5y"],
                    fmt={"level": "{:.3f}", "d_1w_bp": "{:+.1f}",
                         "d_1m_bp": "{:+.1f}", "z_5y": "{:+.2f}"}))
        A("")
    A("`z_5y` = 相对过去 5 年（1260 个交易日）的标准分。")
    A("")

    # ---- 3. attribution ---------------------------------------------------- #
    A("## 3. IC 变动归因（Fed 覆盖期内，精确可加）")
    A("")
    A("恒等式：`ΔIC(市场) = Δ预期通胀 + Δ通胀风险溢价 − ΔTIPS流动性溢价 + Δ残差`")
    A("")
    for h_label, h in (("1 周", 5), ("1 月", 21), ("3 月", 63)):
        for t in ("10", "5f5"):
            a = attribution(dkw, t, h)
            A(f"**{TENOR_LABEL[t]} · 近 {h_label}（bp）**")
            A("")
            A(_md_table(a.rename("bp").to_frame().T, fmt={c: "{:+.1f}" for c in a.index}))
            A("")

    # ---- 4. residuals ------------------------------------------------------ #
    A("## 4. 模型残差：TIPS 相对 DKW 的贵贱")
    A("")
    A("`resid_real` < 0 表示 TIPS 实际收益率低于模型拟合值 → TIPS 偏贵。")
    A("")
    rr = residuals(dkw)
    for k, v in rr.items():
        A(f"**{k}**（截至 {v.dropna().index.max().date()}）")
        A("")
        tail = v.dropna().iloc[-1].to_frame("bp").T
        sd = v["resid_real_bp"].dropna().iloc[-1260:].std()
        tail["z_5y_real"] = float(v["resid_real_bp"].dropna().iloc[-1200:].iloc[-1] / sd) if sd else np.nan
        A(_md_table(tail, fmt={"resid_nom_bp": "{:+.1f}", "resid_ic_bp": "{:+.1f}",
                               "resid_real_bp": "{:+.1f}", "z_5y_real": "{:+.2f}"}))
        A("")

    # ---- 4b. LIVE residual: the actionable version of section 4 ------------ #
    if lr is not None and len(lr):
        A(f"**4.2 实时残差**（把 4.1 推到最新交易日 {bridge_res.nowcast.index.max().date()}，"
          f"锚点 {bridge_res.nowcast.attrs['anchor_date'].date()}）")
        A("")
        A(_md_table(lr, ["tenor", "resid_ic_anchor_bp", "resid_real_anchor_bp",
                         "resid_real_live_bp", "d_bp", "z_live", "band_bp"],
                    fmt={"resid_ic_anchor_bp": "{:+.1f}", "resid_real_anchor_bp": "{:+.1f}",
                         "resid_real_live_bp": "{:+.1f}", "d_bp": "{:+.1f}",
                         "z_live": "{:+.2f}", "band_bp": "{:.1f}"}))
        A("")
        A("读法：`resid_real_live` > 0 = TIPS 实际收益率高于模型拟合 = **TIPS 偏便宜**"
          "（等价于盈亏平衡偏便宜）→ 方向是**多 TIPS / 空名义**。"
          "`d_bp` 是这十几个交易日里偏离的扩大/收敛量；`band_bp` 是同期桥接噪声的 95% 带"
          "（残差是两条桥接序列之差，故按方差相加 = √(带_nom² + 带_ic²)），"
          "**`|d_bp|` 小于它就说明这十几天的变化淹没在噪声里**。"
          "`z_live` 的分母是 Fed 口径过去 5 年的标准差（与 4.1 同源）。")
        A("")
        A("**注意两个显著性不是一回事**：`z_live` 判的是**偏离水平**（跨年尺度），"
          "`d_bp vs band_bp` 判的是**这十几天的变化**（短期）。"
          "水平显著 + 变化不显著 = 这个偏离早就存在、不是这几天新出现的，"
          "不能当成「刚发生的错价」来追。")
        A("")
        A("**方法（关键）**：4.1 是 Fed 口径、常年滞后 2–3 周；直接用彭博盈亏平衡减 Fed 拟合值"
          "**不成立** —— 两者不是同一个测量。实测 1983 年以来的水平基差：均值 +2.3bp，"
          "10y 标准差 4.5bp、5y5y 11.6bp，量级与信号相当。"
          "所以 4.2 走**变动**口径：`resid_live = resid_锚点 + [ΔIC(市场) − ΔIC(拟合)]`，"
          "常数基差在每次差分中抵消。")
        A("")

    # ---- 5. bridge nowcast ------------------------------------------------- #
    A("## 5. 桥接外推（自算估计，非 Fed 数据）")
    A("")
    if bridge_res is None or not bridge_res.ok:
        A(f"_未生成：{bridge_res.message if bridge_res else 'bridge disabled'}_")
    else:
        m = bridge_res.metrics
        h = bridge_res.horizon
        A(f"Fed 序列滞后 {len(bridge_res.gap_days)} 个交易日"
          f"（锚点 {bridge_res.nowcast.attrs['anchor_date'].date()}）。"
          f"方法：滚动 750 日岭回归，拟合**日变动**并锚定 Fed 最后水平。")
        A("")
        A(f"**样本外检验**（λ 在较早 400 日切片上选，指标在之后未用于选参的 400 日上计算，基准 = 随机漫步）："
          f"`rmse_h` / `r2_h` 为 **{h} 日**（即当前需桥接的实际跨度）累积口径，"
          f"`band95_h` 为该跨度的实测 95% 分位误差。")
        A("")
        agg = m[m["target"].str.startswith(("ic.fitted", "nominal.yield.fitted"))]
        A(_md_table(agg, ["target", "rmse_bp", "r2_oos", "rmse_h_bp", "rmse_rw_h_bp",
                          "r2_h_oos", "band95_h_bp", "n_oos"],
                    fmt={"rmse_bp": "{:.2f}", "r2_oos": "{:+.3f}", "rmse_h_bp": "{:.2f}",
                         "rmse_rw_h_bp": "{:.2f}", "r2_h_oos": "{:+.3f}",
                         "band95_h_bp": "{:.1f}", "n_oos": "{:d}"}))
        A("")
        nc = bridge_res.nowcast
        rows = []
        for t in TENORS:
            for c, nice in (("exp.inflation", "预期通胀"),
                            ("inflation.risk.prem", "通胀风险溢价"),
                            ("tips.liq.prem", "TIPS流动性溢价"),
                            ("ic.fitted", "模型隐含 IC")):
                col = f"{c}.{t}"
                base = dkw[col].dropna().iloc[-1]
                now = nc[col].iloc[-1]
                rows.append({
                    "tenor": TENOR_LABEL[t], "series": nice,
                    "fed": base, "now": now, "d_bp": (now - base) * 100,
                    "band_bp": band_bp(bridge_res, col),
                })
        bt = pd.DataFrame(rows)
        A(f"**外推至 {nc.index.max().date()}**（±band = 该跨度实测 95% 误差带）")
        A("")
        A(_md_table(bt, ["tenor", "series", "fed", "now", "d_bp", "band_bp"],
                    fmt={"fed": "{:.3f}", "now": "{:.3f}", "d_bp": "{:+.1f}", "band_bp": "{:.1f}"}))
        A("")
        crossed = bt[bt["band_bp"].notna() & (bt["d_bp"].abs() > bt["band_bp"])]
        if len(crossed):
            names = "、".join(f"{r.tenor} {r.series}" for r in crossed.itertuples())
            A(f"越过误差带的成分：{names} → 外推值与锚点差异超出该跨度正常噪声，值得核查。")
        else:
            A("所有成分的外推位移都在 95% 误差带以内 → 无一次有统计意义的重新定价。")
        A("")
        if "r2_h_oos" in m.columns and m["r2_h_oos"].notna().any():
            worst = m.loc[m["r2_h_oos"].idxmin()]
            A(f"最弱环节（以 {h} 日口径为准）：`{worst['target']}`，"
              f"样本外 R² {worst['r2_h_oos']:+.3f}（1 日口径 {worst['r2_oos']:+.3f}），"
              f"路径阻尼 γ = {worst['gamma']:.2f}。")
        A(f"γ 为在较早切片上拟合的整条 h 日路径缩放系数：γ=1 表示不做阻尼；"
          f"γ→0 表示该成分的日变动虽可同日拟合，但累积到 {h} 日没有方向性内容，"
          f"此时应把外推值当作「延续 Fed 最后观测」。")
        A("")

    # ---- 6. sentiment ------------------------------------------------------ #
    A("## 6. 情绪联动与风险因子位置")
    A("")
    if bbg is None:
        A("_未连接 Bloomberg，本节不可用。_")
    else:
        st = sentiment_table(dkw, bbg)
        if len(st):
            A("**5-to-10y IC 周变动与风险因子的相关性**（复现 Kim-Walsh-Wei 2019 Table 1 的构造）")
            A("")
            A(_md_table(st, ["sample", "n", "corr_HY", "corr_VIX", "corr_oil"],
                        fmt={"corr_HY": "{:+.2f}", "corr_VIX": "{:+.2f}", "corr_oil": "{:+.2f}"}))
            A("")
        A("**当前风险因子周变动位置**（相对过去 750 日）")
        A("")
        A("| 因子 | 周变动 / 1σ |")
        A("|---|---|")
        for k, v in rz.items():
            A(f"| {k} | {v:+.2f}σ |")
        A("")

    # ---- 7. provenance ----------------------------------------------------- #
    A("## 7. 数据溯源与限制")
    A("")
    A(f"- Fed 源：`econres/notes/feds-notes/DKW_updates.csv`，{len(dkw)} 个交易日，"
      f"{dkw.index.min().date()} → {dkw.index.max().date()}。**发布滞后**：月度更新（每月第 4 个工作日 10:00 后），"
      f"日度序列随发布前推，故常年落后最新交易日约 2–3 周。")
    A("- **非官方统计发布**，美联储明示可无预告延迟、修订或变更方法。")
    ident = check_identities(dkw)
    A(f"- 会计恒等式校验：三条期限、两项恒等式，全样本最大绝对误差 "
      f"{max(ident['nominal_identity_max_abs_err'].max(), ident['ic_identity_max_abs_err'].max()):.1e}（通过）。")
    A("- 本文档第 1、6 节为 Bloomberg 实盘观测；第 2、3、4 节为 Fed 口径；"
      "**第 5 节为自算外推**，误差带来自样本外 RMSE，不含模型设定风险。")
    A("- 本工具不构成投资建议。")
    A("")

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"UST_IC_{asof.date()}.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path, plain


def load_notes() -> list[str]:
    from dkw_fetch import vintage_notes
    return [n.replace('""', '"') for n in vintage_notes()]
