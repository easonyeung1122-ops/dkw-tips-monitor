# DKW 分解方法论与校验记录

本文档记录 `dkw-tips-monitor` 的数据口径、恒等式、验证协议与已知限制。
所有校验都可在本机复跑。

---

## 1. 来源

| 项 | 内容 |
|---|---|
| 原始论文 | D'Amico, S., D. H. Kim, and M. Wei (2018), "Tips from TIPS: The Informational Content of Treasury Inflation-Protected Security Prices," *JFQA* 53(1), 395–436 |
| 更新说明 | Kim, D., C. Walsh, and M. Wei (2019), "Tips from TIPS: Update and Discussions," FEDS Notes, 2019-05-21, DOI 10.17016/2380-7172.2355 |
| 数据 | `https://www.federalreserve.gov/econres/notes/feds-notes/DKW_updates.csv` |
| 索引页 | `https://www.federalreserve.gov/econres/economic-research-data.htm` |
| Note 页 | `https://www.federalreserve.gov/econres/notes/feds-notes/tips-from-tips-update-and-discussions-20190521.html` |

⚠️ Note 正文里给的 CSV 链接写作 `DKW-updates.csv`（连字符），**是死链**；
索引页与本机实测可用的都是 `DKW_updates.csv`（下划线）。

---

## 2. CSV 结构

- 头部 14 行是说明文字，第 15 行是列名（`date` 开头），第 16 行起是数据。
- 本机实测：**11,391 行，1983-01-03 → 2026-08-31**，27 个数据列。
- 单位：百分比点。`NA` 出现在 TIPS 上市前（`tips.liq.prem.*` 与 `ic.*`）。

列按 **9 列一块 × 3 个期限**排列：

| 块 | 期限后缀 | 列顺序 |
|---|---|---|
| 1 | `.5` | exp.real.short.rate, exp.inflation, real.term.prem, inflation.risk.prem, tips.liq.prem, nominal.yield.raw, nominal.yield.fitted, ic.raw, ic.fitted |
| 2 | `.10` | 同上 |
| 3 | `.5f5` | 同上（5-to-10-year 远期） |

`raw` = 市场观测值，`fitted` = 模型拟合值。

---

## 3. 恒等式（已全样本验证）

```
nominal.yield.fitted = exp.real.short.rate + exp.inflation + real.term.prem + inflation.risk.prem
ic.fitted            = exp.inflation + inflation.risk.prem − tips.liq.prem
```

`dkw_fetch.check_identities()` 全样本 11,391 行的最大绝对误差：

| 期限 | 名义恒等式 | IC 恒等式 |
|---|---|---|
| 5y | 5.7e-14 | 1.4e-14 |
| 10y | 6.0e-14 | 1.4e-14 |
| 5y5y fwd | 5.7e-14 | 1.2e-14 |

即 CSV 内部完全自洽，可直接当作分解的会计基准使用。

**推论**：`ic.raw − ic.fitted` 这个残差是真实存在的、可计算的量；
`real` 端的残差 `resid_real = resid_nom − resid_ic = (nom.raw − nom.fitted) − (ic.raw − ic.fitted)`。
`resid_real > 0` 表示 TIPS 实际收益率高于模型拟合值 → **TIPS 偏便宜**。

### 8/31/2026 实测

| 期限 | resid_nom | resid_ic | resid_real | z_5y(resid_real) |
|---|---|---|---|---|
| 5y | −0.3 | +2.6 | −2.9 | −0.50 |
| 10y | −1.7 | −13.4 | +11.7 | +2.11 |
| 5y5y fwd | −3.1 | −29.5 | +26.4 | +1.68 |

---

## 4. 桥接外推协议

目标：把 Fed 分解从最后发布日外推到最新交易日（常态 10–15 个交易日空档）。

```
d(component)_t = a + b'·dX_t + e_t        滚动 750 日岭回归
component_t    = anchor + γ · Σ d(component)
```

`dX` = Δ实际收益率(5/10/20/30y)、Δ盈亏平衡(5/10/20/30y)、ΔVIX、ΔHY OAS、Δ油价%、ΔMOVE，共 12 个。
标准化后 winsorise 到 ±5σ（抑制 2020 类极值），截距不惩罚。

**为什么用日变动而不是水平**：两个持续性序列做水平回归会得到 ~0.99 的虚假 R²，
日变动口径下的样本外技能才是诚实的。

**为什么锚定 Fed 最后水平**：Fed 的 print 是权威值，我们只做外延。

### 验证协议（防自欺）

| 环节 | 做法 |
|---|---|
| λ 选择 | 较早 400 日切片，按 1 日 MSE |
| γ 选择 | 较早 400 日切片，按 h 日口径；`γ* = E[pred_h·act_h] / E[pred_h²]`，截尾到 [0,1] |
| 指标报告 | 之后**未用于选参**的 400 日切片 |
| 基准 | 随机漫步（预测变动 = 0） |

γ 的公式推导：`cum_err = Σ(pred − act)` ⟹ `Σpred = cum_err + act`；
误差 `= γ·Σpred − act`；最小化 `E[误差²]` 得 `γ* = E[Σpred·act]/E[Σpred²]`。

⚠️ 别把 γ 乘在误差上（`γ·Σ(pred−act)`）——那等于同时阻尼了实际值，会让 γ 搜索必然收敛到 0。

### 实测技能（2026-09-17，h = 13 个交易日，样本外 400 日）

| target | 1日 RMSE | 1日 R² | h日 RMSE | h日 RW RMSE | **h日 R²** | band95 |
|---|---|---|---|---|---|---|
| ic.fitted.5 | 1.30 | 0.742 | 3.03 | 9.49 | **0.898** | 6.3 |
| ic.fitted.10 | 0.92 | 0.799 | 2.24 | 7.29 | **0.906** | 4.7 |
| ic.fitted.5f5 | 0.75 | 0.832 | 2.21 | 6.11 | **0.870** | 4.2 |
| nominal.yield.fitted.10 | 1.66 | 0.883 | 2.38 | 13.94 | **0.971** | 5.0 |
| exp.inflation.10 | 0.39 | 0.877 | 0.61 | 3.18 | **0.963** | 1.25 |
| tips.liq.prem.10 | 0.98 | 0.667 | 2.17 | 6.24 | **0.878** | 4.75 |

最弱成分始终是 `tips.liq.prem`（模型最不市场化的一块），但 h 日口径仍显著优于随机漫步。

### 误差带

必须用验证期**实测 h 日误差的 95% 分位**（`band95_h_bp`）。
**不要**用 `1.96 × 1日 RMSE × √h`：桥接误差高度自相关，√h 缩放会低估带宽约一个数量级。

---

## 5. 市场侧（零模型）口径

第 1 节不依赖任何模型：

```
real_implied = nominal − breakeven        （定义式，保证 Δ名义 = Δ实际 + Δ盈亏平衡恒等）
basis_tii    = real_tii_ticker − real_implied   （报价源差异，不是信号）
```

远期由两点复利自算：`fwd = [(1+y_long)^n_long / (1+y_short)^n_short]^(1/n) − 1`。
提供 5y5y（5y/10y）、10y10y（10y/20y）、20y10y（20y/30y）。

窗口按**工作日**计数：h ∈ {1, 5, 21, 63}。美假日由 Bloomberg `nonTradingDayFillOption=NON_TRADING_WEEKDAYS`
+ `PREVIOUS_VALUE` 填充后保留，以对齐 Fed 在全部工作日发布的口径 —— 代价是 h=21 的实际交易日数可能少 1 天。

**展示时的口径纪律**：`level_*` 与 `basis_tii` 是水平量，**与窗口无关**，因此只在快照表里出现一次；
`Δ` 随窗口变化，做成"行 = 度量 × 窗口、列 = 期限"的变动矩阵。把水平量复制进每个窗口的表里
会让人误以为表格出错（实际踩过）。

`Δ名义 = Δ实际 + Δ盈亏平衡` 在**每一行**都恒等成立（因为 `实际 ≡ 名义 − 盈亏平衡` 是定义式），
实测最大残差 3.6e-15 bp，可直接当回归测试用。

## 6. 实时残差（报告 4.2）：为什么必须走变动口径

第 4.1 节的残差是 Fed 口径、滞后 2–3 周，而它是全篇唯一带 z 分、可直接用于判断的输出来源。
把它推到最新交易日有两条路，只有一条成立：

```
(a) 水平口径（错）  resid_live = ic.raw(bbg) − ic.fitted(bridged)
(b) 变动口径（对）  resid_live = resid_锚点 + [ΔIC(市场) − ΔIC(拟合)]
```

(a) 不成立的原因：彭博反推的 IC（`nom − real_TII`）与 Fed 的 `ic.raw` **不是同一个测量**。
实测 1983 年以来的水平基差：均值 +2.3bp，5y sd 5.6bp、10y sd 4.5bp、5y5y sd 11.6bp ——
与信号（10y 偏离 11.7bp）同量级，直接用水平相减等于把测量差当成错价。
(b) 里常数基差在每次差分中抵消，所以正确。

**残差**本身也不是 ticker 对比，而是 Fed 自己的会计恒等式：

```
resid_nom  = nominal.yield.raw − nominal.yield.fitted
resid_ic   = ic.raw            − ic.fitted
resid_real = resid_nom − resid_ic        ← 因为 nominal ≡ real + IC
```

**符号**：`resid_real > 0` = 市场实际收益率高于模型拟合 = TIPS 价格偏低 = **TIPS 偏便宜**
（等价于盈亏平衡偏便宜）→ 方向是**多 TIPS / 空名义**。

**误差带**：`resid_real` 是两条独立桥接序列之差，所以 95% 带是
`√(band_fitted_nom² + band_fitted_ic²)`，不是其中任一条。只用一条会低估约 30%。

**两种显著性要分开读**：

| 量 | 判什么 | 尺度 |
|---|---|---|
| `z_live` | 偏离**水平**是否异常 | 跨年（分母 = Fed 口径 1260 日 sd） |
| `d_bp` vs `band_bp` | 这十几天偏离**变动**是否超噪 | 短期 |

水平显著 + 变动不显著 = 偏离早已存在，**不是刚发生的错价**，追进去要承担它继续走阔的风险。


Bloomberg ticker（全部经 blpapi 实测有效）：

| 类别 | ticker |
|---|---|
| 名义 | USGG2YR / USGG3YR / USGG5YR / USGG7YR / USGG10YR / USGG20YR / USGG30YR Index |
| TIPS 实际 | USGGT05YR / USGGT10YR / USGGT20YR / USGGT30YR Index |
| 盈亏平衡 | USGGBE05 / USGGBE10 / USGGBE20 / USGGBE30 Index |
| 通胀互换 | USSWIT5 / USSWIT10 Curncy |
| 风险 | VIX Index、LF98OAS Index、MOVE Index、CO1 Comdty（Brent）、CL1 Comdty（WTI 参考） |

**无效**：`USGG15YR`、`USGG15`、`USGGT5YR`、`USGGT7YR`、`USGG5F5`、`USGGBE5`、`USGGR5/R10`、`H0A0`、`BZ1`。

---

## 7. 情绪联动（复现原 Note Table 1）

原 Note 用 5-to-10-year IC 的**周变动**与 HY 利差、VIX、油价（同为周变动）求相关。
本工具同构造复跑（注意：模型 vintage 已不同于 2019 原文，数值只能同向量级比较）：

| 样本 | corr(HY) | corr(VIX) | corr(油价) |
|---|---|---|---|
| 全样本 1999– | −0.26 | −0.17 | +0.17 |
| 危机后 2009/7– | −0.28 | −0.27 | +0.22 |
| 危机前 –2008/7 | −0.20 | +0.02 | +0.11 |

与原文（全样本 −0.33 / −0.22 / +0.18；危机前 VIX 近 0）**符号与量级一致**，
支持原文结论：危机后长端 IC 与风险情绪联动更强，危机前 VIX 相关几乎为零。

---

## 8. 已知限制

1. **vintage 漂移**：当前 CSV 的模型设定 ≠ 2019 FEDS Note。已改用 BCFF 月度 1 年通胀预测，
   并加入 15y/20y 名义与 TIPS 收益率参与估计；参数基于固定再估样本外推。
   引用任何数值前先看 `vintage_notes()`。
2. **发布滞后 2–3 周**，第 2–4 节永远不是最新。要最新只能用第 1 节（零模型市场侧）或第 5 节（自算）。
3. **第 5 节含模型设定风险**：误差带只反映参数估计误差（来自样本外 RMSE），
   不含"ridge + 12 个市场因子"这一函数形式本身可能是错的这一风险。
4. **期限上限 10 年**：Fed 分解只有 5y / 10y / 5f5。20y / 30y 只有市场侧（第 1 节），无成分分解。
5. **模型残差不等于套利机会**：`resid_real` 反映 TIPS 相对模型的偏离，
   可能来自未建模的流动性、供需、指数化债特有的税收/通胀意外，不必然是方向性交易。
6. **无官方背书**：Fed 明示该发布非官方统计，可无预告延迟、修订或变更方法。
7. **本工具不构成投资建议。**

---

## 9. 第 0 节「大白话版」：规则与映射

实现在 `analyze.plain_read()`，**每次运行必出**，位置在结论行之后、第 1 节之前；`run.py` 同时打到 stdout（`--terse` 只抑制打印）。

**性质**：纯派生函数。输入全部是本次运行已算好的中间量（`label`、`attr_gap`、`market_table`、`rz`、`live_residual`），
**不接受新数据、不另立口径**。若它与下方表格冲突，表格是对的，`plain_read()` 是 bug。

**叙述顺序 = 决策顺序**（不跳步）：

| # | 内容 | 数据来源 | 关键换算 |
|---|---|---|---|
| 1 | 时间差与幅度 | `gap_days`、`ic.raw.10`、`be_10y` | bp，并明确"1 个基点 = 0.01%" |
| 2 | 名义收益率是谁在推 | 1.2 的 10y 近 1 月（缺则近 1 周） | `Δ实际/Δ名义` → "约 X% 的波动来自实际利率" |
| 3 | 通胀预期本身动了吗 | `classify()` 的判定标签 + 归因三项 | 见下表 |
| 4 | 有没有错价 | 4.2 的 `resid_real_live_bp` / `z_live` / `d_bp` / `band_bp` | `z` → "几倍平常波动"；`d_bp vs band_bp` → "是不是这几天新出现的" |
| 5 | 一句话总结 | 第 2、4 步的结论 | — |

**判定标签 → 白话映射**（避免把统计口径说成预测）：

| 标签 | 白话落脚点 |
|---|---|
| 模型残差主导 | 「读不出来」——变动落在模型解释不了的地方（TIPS 供需/流动性），既不是通胀看法变了也不是风险补偿变了 |
| 预期驱动 | 「通胀预期确实动了」，并说明该渠道未被抵消 |
| 情绪/溢价驱动 | 「动的是风险补偿，不是对通胀的看法」；若风险因子周变动 > 0.5σ，补一句是情绪在推 |
| 混合 | 「几个渠道都有份，没有谁占主导，读不出干净结论」 |
| 中性 | 「净变动比日常噪声还小，等于没动」 |

**显著性必须说两面**：`z_live ≥ 2` 只说明偏离**水平**不小；还要报 `|d_bp|` 是否超过 `band_bp` ——
未超过时明确写「早就存在、不是这几天新冒出来的，追进去要先扛浮亏」。这一条是第 0 节最容易被写坏的地方：
把"水平显著"讲成"刚出现的机会"，等于制造交易信号。

**格式硬规则**：日期走 `_d()`（否则渲染成 `2026-08-31 00:00:00`）；带符号数字走 `_num()`（复用 `_neg_zero()`，避免 `-0.0`）；
术语只在括号里就地翻译，段尾统一给"基点 / z 值"的白话对照。措辞只改 `plain_read()`，不要手改生成的 `.md`。

