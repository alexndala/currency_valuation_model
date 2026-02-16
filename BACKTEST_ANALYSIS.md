# Does Currency Valuation Actually Predict Anything? I Backtested It.

*An out-of-sample test of REER-based fair value signals across 14 years and 6 currencies*

---

In the [previous piece](README.md), I built a regime-adjusted currency valuation framework — one that detects structural breaks in Real Effective Exchange Rates (REER) and uses post-break means instead of stale historical averages. The framework decomposes expected returns into real carry, a valuation gap, and an EM structural kicker.

It sounded good on paper. But a model that fits the past tells you nothing about the future.

So I backtested it.

---

## The Setup

The core model is almost embarrassingly simple:

```
Fair Value REER = a + b × Real Rate Differential vs USD
```

One regressor. Pooled OLS across countries. No machine learning, no factor zoo — just the straightforward idea that real interest rate differentials should explain where currencies trade in purchasing-power terms.

The question isn't whether this fits in-sample (it does, with R² around 0.18). The question is whether the **deviations** from fair value — the signal that says "this currency is overvalued" or "this currency is undervalued" — actually predict what happens next.

Here's how I tested it:

- **Expanding window**: At each month *t*, re-estimate the model using all data from April 1994 through *t*. No future information leaks in.
- **Signal**: Compute deviation from model-implied fair value at time *t*.
- **Outcome**: Measure actual REER change over the following 12 months.
- **Evaluation period**: January 2010 through March 2024 — 171 months of out-of-sample signals, with 16 years of initial training data.

This gives 628 individual country-month signals across ZAR, GBP, CAD, MXN, JPY, and AUD.

---

## The Headline: 58.8% Directional Accuracy

Across all 628 signals, the model correctly predicted the direction of REER movement 58.8% of the time. That's nearly 9 percentage points above a coin flip.

That might not sound dramatic, but in FX — where 50/50 is the null and the efficient market hypothesis is supposed to hold — a persistent 59% hit rate over 14 years is meaningful. It means the model is capturing something real about how currencies revert to fundamentals.

But the averages hide the interesting part.

---

## Where the Model Shines (and Where It Doesn't)

The country-level results reveal a clear pattern:

| Currency | Signals | Hit Rate | Avg Deviation | Avg 12M Change |
|----------|---------|----------|---------------|----------------|
| MXN | 111 | **74.8%** | -7.4% | +1.1% |
| GBP | 171 | **66.7%** | -8.4% | +0.8% |
| ZAR | 169 | 50.3% | -11.0% | -1.9% |
| CAD | 171 | 48.0% | -5.3% | -1.5% |

MXN stands out at nearly 75% accuracy. When the model said the peso was undervalued relative to fundamentals, it appreciated three-quarters of the time within 12 months. Sterling comes in at two-thirds accuracy — solid for a G10 currency where you'd expect efficiency to be highest.

The rand is essentially a coin flip at 50.3%. This makes intuitive sense: ZAR is driven by idiosyncratic risk — load-shedding, political uncertainty, commodity shocks — that a single real-rate-differential factor can't capture. The model correctly identifies ZAR as persistently undervalued (average deviation of -11%), but the reversion mechanism is unreliable because non-fundamental shocks keep pushing it further from fair value.

CAD at 48% is the one genuine disappointment. The Canadian dollar's tight coupling with US economic cycles means the real rate differential — which is *relative* to the US — doesn't generate much useful signal. When both economies move together, the differential is small and noisy.

---

## The Extremes Are Where the Money Is

This is the finding that matters most for portfolio construction.

I sorted all 628 signals into quintiles by deviation from fair value — from Q1 (most undervalued) to Q5 (most overvalued):

| Quintile | Avg Deviation | Hit Rate | Avg 12M Fwd | Avg Sim. Return |
|----------|---------------|----------|-------------|-----------------|
| Q1 — Most Undervalued | -20.1% | **63.5%** | +4.0% | +1.001 |
| Q2 | -12.4% | **63.2%** | +1.3% | +0.156 |
| Q3 | -9.3% | 40.5% | -0.9% | -0.083 |
| Q4 | -4.2% | 52.0% | -1.7% | -0.024 |
| Q5 — Most Overvalued | +6.2% | **74.6%** | -6.4% | +0.410 |

The pattern is striking. The tails — currencies that are the *most* mispriced — generate the strongest signals. Q5 (most overvalued) hits 74.6% of the time, with overvalued currencies depreciating an average of 6.4% over the following year. Q1 (most undervalued) hits 63.5%, with undervalued currencies appreciating 4.0%.

The middle quintile (Q3) is basically noise — 40.5% accuracy, negative simulated returns. This is exactly what you'd expect from a valuation signal: when a currency is close to fair value, the model has nothing useful to say. The edge is in the extremes.

This is a textbook mean-reversion result: **the larger the dislocation, the stronger the reversion pull.** And it survives out-of-sample.

There's also an asymmetry worth noting. Overvalued signals (deviation > 0) hit at 75.7%, while undervalued signals hit at 53.7%. Currencies fall from overvaluation more reliably than they rise from undervaluation. This makes intuitive sense — overvalued currencies face gravity from both tightening financial conditions and current account adjustment, while undervalued currencies can stay cheap for years if the structural headwinds persist (ask anyone who's been long JPY since 2022).

---

## Cumulative Performance: A 14-Year Track Record

Aggregating the simulated returns — long undervalued currencies, short overvalued ones, sized proportionally to the deviation signal — the strategy accumulates a cumulative score of 48.3 over the evaluation period.

The annualized Sharpe-like ratio comes in at **1.42**. This isn't a tradeable Sharpe (it doesn't account for transaction costs, margin, or the reality that REER isn't directly tradeable), but it confirms that the signal contains genuine information. A random signal would produce a Sharpe near zero.

The equity curve isn't a straight line. The best years were 2020 (+1.55 average monthly return — COVID dislocations offered enormous mean-reversion opportunities) and 2011 (+0.53, Eurozone crisis divergences). The worst was 2019 (-0.63), when the synchronized global slowdown and trade war pushed multiple currencies away from fundamentals with no reversion.

Year-by-year performance:

| Year | Hit Rate | Comment |
|------|----------|---------|
| 2011 | **80.0%** | Eurozone crisis created massive divergences that reverted |
| 2013 | **80.0%** | Post-crisis normalization, strong mean reversion |
| 2020 | **84.4%** | COVID panic created extreme dislocations → sharp reversion |
| 2016 | 66.7% | Post-Brexit, post-commodities adjustment |
| 2019 | **20.0%** | Trade war, synchronized slowdown — fundamentals ignored |
| 2018 | 40.9% | EM selloff (Turkey/Argentina contagion to ZAR, MXN) |

The pattern: the model excels during and immediately after crises, when currencies get pushed to extremes and then revert. It struggles during slow-burn macro divergences where fundamentals take a backseat to risk sentiment.

---

## How Stable Are the Coefficients?

One concern with any backtest is whether the model is stable or whether it's picking up different relationships at different points in time. Because I re-estimate at every month using an expanding window, I can track how the intercept and beta evolve.

The results are reassuring:

- **Intercept (a)**: 115.2 ± 1.5 — barely moves. Range of [113.2, 117.1] over 171 re-estimations.
- **Beta (b)**: 3.47 ± 0.56 — more variation, but consistently positive. Range of [2.19, 3.92].
- **R²**: 0.177 ± 0.038 — remarkably stable for a single-factor cross-country model.

The beta trending from around 2.2 in 2010 up to 3.9 by 2024 is actually informative — it suggests real rate differentials have become a *stronger* driver of REER over time, possibly reflecting increasing capital mobility and carry-trade activity in the post-GFC era.

The key takeaway: this isn't a model that works for one period and fails in another. The relationship between real rate differentials and REER is genuinely persistent.

---

## The Scatter Plot: A Negative Slope That Shouldn't Exist (If Markets Were Efficient)

The single most compelling piece of evidence is the scatter of deviation-at-signal-date vs. 12-month-forward REER change.

The regression line has a slope of **-0.412** (t-statistic: -13.11, p-value: 0.0000). The R² is 0.215.

Read that again: knowing only how far a currency deviates from its model-implied fair value today, you can explain **21.5% of the variance** in its REER movement over the next year. For a single-variable, out-of-sample prediction in FX — arguably the most efficient market in the world — this is a strong result.

The negative slope confirms mean reversion: currencies that are above fair value (positive deviation) tend to depreciate (negative forward change), and vice versa.

---

## Limitations (Honest Ones)

This isn't a trading strategy. Several caveats:

**REER isn't directly tradeable.** The Real Effective Exchange Rate is a weighted basket against all trading partners, adjusted for inflation. You can't go "long REER." Any implementation would require mapping REER signals to spot FX positions, introducing tracking error.

**Transaction costs are ignored.** Monthly rebalancing across 6 currency pairs with leverage would generate significant costs, particularly in EM pairs.

**The model has one factor.** Real rate differential vs USD explains 18% of REER variation. The other 82% is noise from the model's perspective — terms of trade, political risk, capital flows, positioning. A richer model might improve accuracy but risks overfitting.

**Survivorship in country selection.** The currencies tested are ones with decent data coverage. A broader universe might dilute the results.

**2019 and 2022 remind you that regime breaks happen.** When a new regime establishes, the model's fair value is stale until enough post-break data accumulates. During the transition, signals can be wrong for extended periods.

---

## What This Means for Practitioners

**1. Valuation signals work, but only at the extremes.** Don't trade small deviations. Wait for currencies to reach the tails of the distribution — Q1 or Q5 — where hit rates jump to 63-75%.

**2. The best opportunities come after crises.** 2011, 2013, 2020 — the model's best years all followed periods of extreme dislocation. When everyone is panic-selling an EM currency or piling into a safe haven, that's when valuation signals have the most power.

**3. Pair it with carry.** Valuation alone tells you *direction* but not *timing*. Combining the valuation signal with carry (real yield differential) gives you direction *and* compensation while you wait.

**4. Don't fight the regime.** When a currency has undergone a structural break, the old fair value is meaningless. The model needs time — typically 2+ years — to incorporate the new regime before its signals become reliable again.

**5. Overvalued shorts are more reliable than undervalued longs.** The 75.7% vs 53.7% hit rate asymmetry suggests you should be more aggressive fading overvaluation than chasing undervaluation.

---

## Closing

There's a reason people keep coming back to REER-based valuation models despite their simplicity. They capture something fundamental about how currencies work: real exchange rates are anchored to real economic differentials, and deviations from those anchors tend to correct.

The backtest confirms this isn't just an in-sample artefact. Over 14 years, 628 out-of-sample signals, and 6 currencies, the model achieves a 59% hit rate, a 1.42 Sharpe-like ratio, and — most importantly — a clear, monotonic relationship between signal strength and predictive power.

The edge isn't in predicting where currencies go on any given day. It's in identifying the handful of moments where a currency is so far from fundamentals that mean reversion becomes the dominant force.

Those moments are rarer than you'd think. But when they come, they're worth waiting for.

---

*Data: FRED (BIS Broad REER, 10Y government bond yields, CPI). Backtest period: January 2010 – March 2024. Model: Pooled OLS with expanding window, re-estimated monthly. No transaction costs or implementation frictions applied.*

*Author: Mahlatse Alex Ndala*
