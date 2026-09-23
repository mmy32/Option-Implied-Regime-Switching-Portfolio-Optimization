I've assumed the field is empirical finance / asset pricing. If you meant a different target field (e.g. operations research), tell me and I'll re-weight the comments.

**Abstract**

[Major] "outperforming an unpenalized mean-variance baseline, a static 60/40 allocation, and a naive VIX threshold heuristic." The 60/40 and VIX results appear only as lines in Figure 3. No table reports them. Add both to Table 2.

[Major] "annualized return of 37.09% with a Sharpe ratio of 1.57." 37.09/23.69 = 1.566, so the Sharpe ratio is computed on raw returns with no risk-free rate subtracted. This contradicts Section 3.2, which says T-bills are used for excess returns. The same applies to the other rows: 47.56/32.21 = 1.48 and 8.38/20.84 = 0.40. Recompute on excess returns. Also state whether returns are net of transaction costs and whether they are arithmetic or log.

[Minor] "significantly outperforming." No significance test is reported. Either add one (Ledoit–Wolf 2008 bootstrap for Sharpe differences) or drop "significantly."

**1 Introduction**

[Minor] "the agent continuously observes both equity returns and option-implied indicators of volatility and skewness." Skewness does not appear to enter the HMM or the optimizer (see 3.1). Also, "sequential decision process" overstates the model: this is a myopic single-period MVO re-solved weekly, with no dynamic programming or hedging demand. Rephrase.

[Style] Paragraphs 2–3 repeat the motivation in Section 2.1 almost point for point. Cut to two sentences and a forward reference.

**2 Literature Review**

[Minor] The citations don't match the references. "Christoffersen and Chang (2009)" is Chang, Christoffersen, Jacobs and Vainberg. "Ang and Timmermann (2012)" is listed as 2011. Halko et al. is cited as 2011 but listed as 2010. The SIAM Review version is 2011, so cite that.

[Minor] Section 2.3 argues that regime identification is sensitive to the number of states and the filtering method. The paper then never varies either. Either test this (see extensions) or drop the point.

[Style] The reference list is duplicated after Appendix A.

**3 Data and Feature Engineering**

[Major] "2,339 individual equities that were constituents of the S&P 500 index at any point." The active-asset mask only checks that a stock is trading, not that it is in the index at time t. So stocks that later entered the index because they performed well are investable before inclusion. That is a forward-looking selection bias, and it is not removed by keeping delisted firms.

Figure 5 fits this concern. The top holdings include TPL, FIX, ODFL, URI, TDG and AVGO, several of which joined the S&P 500 late in the sample. They are held at the 5% cap from about 2010 onward. Fix: use point-in-time constituent membership and restrict the investable set to index members at t.

[Minor] The sample period is inconsistent: "stocks within the S&P 500 from 2005 to 2025" versus "January 2005 to April 2026." Reconcile.

[Minor] Features are listed but unused. Skewness is described as "proxied using 25-delta risk reversal quotes," yet the text also says the authors pivoted to SKEW because option data were incomplete. The same goes for "option bid-ask quotes and open interest/volume data to filter for liquidity," which has no use when the inputs are indices. State exactly which features enter the HMM.

[Minor] "IV Term Structure Slope: spread between 1-month and 3-month implied volatility" carries essentially the same information as the VIX/VIX3M ratio. Drop one or justify keeping both.

[Minor] Question for the author: VIX3M (formerly VXV) history begins around December 2007. How was the HMM trained on 2005–2008 using this ratio?

[Minor] VRP = IV² − RV. State the units: VIX is in annualized percent, while RV comes from daily returns. State the realized-variance window too. Table 1 reports a higher VRP in stress, whereas VRP usually compresses or turns negative when realized variance spikes. Please confirm the construction.

[Minor] Section 3.2 says per-stock bid-ask spreads and ADV were collected, but λ uses only the SPY spread. Either use the per-stock data or remove the claim.

[Major] The 60/40 benchmark has no bond data source. Specify the bond index.

[Minor] SPX and NDX total-return indices are collected but never used as benchmarks. A cap-weighted SPX total-return benchmark is the obvious comparison and should be added.

**4.1 Alternative Methodologies**

[Minor] "Our methodology overcomes these limitations." Neither Black–Litterman nor risk parity is tested, so this is unsupported. Either add them as benchmarks or cut the subsection to a few lines.

[Style] The risk parity critique ("highly leveraged fixed-income and equity sleeves") doesn't apply to the paper's all-equity setting.

**4.2 HMM**

[Major] "given the inferred regime k." The optimizer uses a hard regime label, but Section 5.2 credits the "nuanced, continuous probability states of the Hidden Markov Model." Pick one. Either use probability-weighted μ and Σ, or remove the claim in 5.2.

[Minor] Several model details are missing:
- the number of states K (two, implied)
- the exact feature vector X_t
- whether the regime at t comes from the filtered probability P(Z_t | X_1..t) or from Viterbi or smoothed decoding

[Minor] Figure 1 caption: are the shaded states the real-time, as-of-t classifications, or the final model applied retrospectively? Only the former supports "out-of-sample."

[Minor] The flat, straight VIX/VIX3M segment around 2009–2010 in Figure 1 looks like interpolated or missing data. Please explain.

[Minor] "anticipating... the 2008 Global Financial Crisis." The figure shows contemporaneous classification, not anticipation. Show a lead-lag analysis (regime probability versus forward drawdowns or forward realized volatility) or drop "anticipating."

[Minor] Table 1: state whether these statistics are in-sample or out-of-sample, and report the number of days in each regime.

**4.3 Covariance Cleaning**

[Major] Look-ahead in μ_k and Σ_k. The expanding window is described for the HMM only. Equation (3) does not say that μ_k and Σ_k are estimated only on data before t.

The results are hard to believe without this. A 37% annualized return over 18 years compounds to roughly 300x, matching Figure 3. The same ten names are held at the cap almost continuously from 2010 to 2026 (Figure 5). Both fit full-sample means. The authors must confirm, with code-level detail, that μ_k and Σ_k use only returns up to t−1.

[Major] Rank-10 truncation with no residual diagonal. A rank-10 approximation of a 2,339×2,339 matrix treats idiosyncratic variance as zero, so the optimizer sees stock-specific risk as free. This pushes the solution to concentrate at the caps. Use a factor model, Σ = BB′ + D, with a diagonal of residual variances, or compare against Ledoit–Wolf shrinkage. Justify r = 10 as well.

[Minor] How are missing returns handled in Σ̂_k for stocks with partial histories? The stress regime likely has T_k far smaller than N.

[Major] μ_k is a sample mean over 2,339 stocks. With long-only weights and a 5% cap, the optimizer essentially picks the top 20 names by sample mean. That is exactly the error maximization the paper claims to prevent. Consider shrinking μ toward a common mean or a factor-implied mean.

**4.4 Optimization**

[Major] The constraints Σw = 1, w ≥ 0 over an all-equity universe mean the portfolio is always 100% invested in equities. The regime switch therefore cannot reduce market exposure. This undercuts the drawdown-protection narrative in 5.3. Either add a cash/T-bill or bond asset, or reframe the regime effect as cross-sectional tilting.

[Minor] Specify the scale of λ (the SPY spread in what units, mapped to λ how), the return frequency behind γ = 2, and whether realized costs are deducted from returns in the backtest. With an SPY spread of about 1bp, λ may be close to non-binding (see 6.1).

**5 Results**

[Major] "benchmarked against four alternatives." Table 2, labelled "Comprehensive," shows only three strategies. Add 60/40, the VIX heuristic, and SPX total return.

[Major] "the L1 penalty acted as an effective regularizer... resulting in a dramatic reduction in volatility." The penalty is on turnover, not on risk. There is no reason it should cut volatility from 32% to 24%, and no ablation isolates it. The two portfolios probably differ in holdings for other reasons. Report turnover, number of holdings, and returns net of costs for both. A 1.57 versus 1.48 Sharpe difference also needs a significance test.

[Major] Figure 2 caption: "suppresses erratic volatility spikes compared to the naive benchmark." The figure shows the L1 portfolio with higher rolling volatility than 1/N for most of 2012–2026, and Table 2 agrees (23.69% vs 20.84%). The caption contradicts the data.

[Major] 5.3: "By shifting its risk model to Σ1,clean during stress regimes, it successfully limited maximum drawdown." This is causal language without an ablation. Given the fully-invested constraint, lower drawdown is more plausibly driven by stock selection. Test it: run the same pipeline with a single regime and compare drawdowns.

[Minor] Figure 4 shows the L1 portfolio with deeper drawdowns than 1/N in roughly 2019 and 2022. Acknowledge this.

[Major] 5.4: "Figure 5 visually confirms this behavior [sparsity]." The top 10 names sum to 50% at the 5% cap. That shows the cap binding, not sparsity. Report the number of non-zero positions, HHI, and turnover over time.

[Major] 5.4: "unless a regime shift dictates a structural realignment." Figure 5 shows near-identical weights through every regime change. That suggests the regime model barely affects allocation, which contradicts the paper's core premise.

[Minor] The gap in Figure 5 during 2009–2010 (top-10 weight near zero) is unexplained.

**6 Robustness**

[Major] "average weekly portfolio turnover drops logarithmically." There are three points, and turnover falls only about 16% (0.108 to 0.091) for a 10x cost increase. That suggests the penalty is weakly binding, not "a strict barrier."

[Minor] The y-axis label ("%") is ambiguous: is 0.10 a 0.1% or a 10% weekly turnover? Also specify which three-year window was used and report net performance at each multiplier. Turnover alone is not the relevant outcome.

[Minor] A single cost stress test is thin for a "robustness" section. See the extensions below.

**7 Limitations**

[Major] The section omits the constituent look-ahead bias and possible μ/Σ look-ahead, which are the most serious limitations.

[Style] The four paragraphs on single-name option data could be one.

[Minor] The trading assumptions are also missing: executing at the close on the same day the signal is computed from closing data. If so, lag execution by one day.

**8 Conclusion**

[Major] "effectively anticipates market stress in real-time" and "exceptional out-of-sample risk-adjusted returns" are unsupported until the look-ahead issues are resolved and anticipation is tested.

[Style] "validates a robust, end-to-end architecture." Replace with a specific claim.

**Appendix A**

[Minor] Equation (9) misstates the Halko–Martinsson–Tropp bound. The actual bound holds in expectation or with a failure probability, and its constant depends on the oversampling p, the rank, N, and the number of power iterations. It is not a generic (1+ε). Give the correct statement and the chosen p and q.

**Suggested extensions**

- Ablations: single-regime versus HMM; sample covariance versus Ledoit–Wolf versus factor-model covariance; λ = 0 versus λ > 0 with identical cost accounting.
- Factor attribution: Fama–French 5 plus momentum alpha. The holdings look like a momentum/quality tilt that may fully explain the returns.
- Multiple testing: deflated Sharpe ratio (Bailey and López de Prado) given the choices of γ, rank, cap and K.
- Sensitivity to K ∈ {2, 3}, rank r, cap size, and rebalance frequency.
- Regime-conditional and subperiod performance, including post-2012 only.
- Realistic per-stock costs using the spread and ADV data already collected.
- A cash or bond sleeve so the regime signal can actually change market exposure.

**Three most important changes before submission**

1. Eliminate look-ahead bias. Use point-in-time S&P 500 membership for the investable universe, and confirm or fix that μ_k and Σ_k are estimated only from data before t. Then rerun everything. The current returns are not credible until this is done.
2. Fix the evidence behind the causal claims. Report all benchmarks in Table 2, net of costs, with excess-return Sharpe ratios and significance tests. Run ablations isolating the HMM, the SVD and the L1 penalty, and add factor-model alpha.
3. Repair the model and the text–figure contradictions. Add an idiosyncratic diagonal to the rank-10 covariance and allow a non-equity asset so regimes can change market exposure. Then correct the claims contradicted by the paper's own evidence: Figure 2's caption, "four alternatives," the sparsity claim in 5.4, "logarithmic" turnover, and "continuous probability states."