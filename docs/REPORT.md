# Leakage-proof evaluation for financial machine learning

Workshop note accompanying the `finmlcv` library (v0.1.0, 2026).

Hossein Tabasi

This note is a companion to the repository README. It records how purged *k*-fold, combinatorial purged cross-validation (CPCV), embargo, and the deflated Sharpe ratio (DSR) are implemented; why shuffled *k*-fold is not a valid evaluator for overlapping financial labels; and what the TOY experiment actually measured. No live trading result is claimed. Where a number appears, it is copied from `results/tables/` after `make toy`.

## 1. The evaluation problem

A supervised learning pipeline in finance typically looks as follows. From a price path one constructs a label \(y_t\) that occupies a *time interval* \([t, t_1]\): a fixed-horizon forward return, a triple-barrier first-touch, or a meta-label on a primary bet (López de Prado 2018, ch. 3). Features \(X_t\) are intended to be measurable at \(t\). A classifier \(\hat f\) is scored by cross-validation, and a long/short overlay of the predictions is summarised by a Sharpe ratio.

Two independent defects hide in that sentence.

The first is **split leakage**. If \(t_1 > t\), then \(y_t\) and \(y_{t+1}\) share the underlying path. A shuffled partition of rows places a non-negligible fraction of those overlapping pairs on opposite sides of the train/test cut. A model that interpolates in a time-local feature space — *k*-nearest neighbours, a deep tree, a network with a positional channel — then predicts the test label from a neighbour whose own label was computed on the same bars. The CV score is a measure of smoothness of the labelled path, not of a decision rule that could have been run at \(t\).

The second is **look-ahead in \(X\)**. If a column of the design matrix is a function of prices after \(t\) (a centred rolling statistic, a merge on a raw timestamp, the forward return itself), then *every* splitter, including purged CV, will look skilled. Purging does not inspect columns. That is the job of an auditor.

A third, older defect sits on top of both: **selection bias** on the backtest. The Sharpe that is reported is usually the best of many configurations. Bailey & López de Prado (2014) give a closed-form haircut, the deflated Sharpe ratio, under an independent-trials approximation. Harvey, Liu & Zhu (2016) document the scale of the problem in the empirical-asset-pricing literature; Arnott, Harvey & Markowitz (2019) state a protocol for backtests in the era of machine learning. White (2000) and Romano & Wolf (2005) are the econometric ancestors.

This library implements the first two checks as code and the third as a function of an *assumed* `n_trials`. It does not estimate `n_trials` from a research log, and it does not know about correlated trials.

## 2. Label lifetime

Let the observation index be \(i = 0, \ldots, n-1\) in time order. A label that starts at \(i\) and ends at \(t_1[i]\) (inclusive) occupies the closed interval \([i, t_1[i]]\). Adjacent labels overlap when \(t_1[i] \ge i+1\), which is automatic for any horizon of at least one bar if labels are sampled every bar.

Triple-barrier labels (AFML ch. 3) make the lifetime stochastic: the vertical barrier is a maximum holding period, the horizontal barriers are profit-taking and stop-loss. The first touch of any of the three ends the label. Meta-labeling conditions on a primary side \(s_t \in \{+1,-1\}\) and replaces the ternary outcome with a binary “the bet worked” label. In all three cases the object that must not straddle a train/test cut is the interval, not the row index.

A practical consequence: dropping NaNs *after* splitting changes which rows land in which fold and can re-introduce overlap that the splitter thought it had removed. `finmlcv.experiments.run_benchmark.align_xy` drops and then remaps \(t_1\) onto the aligned row axis *before* any splitter is constructed.

Survivorship is a different hazard. A panel that includes only names still alive at the end of the sample gives every training fold a peek at who will survive into the test fold. This repository does not build a point-in-time universe; a supervisor who extends the FULL path to CRSP-style data should.

## 3. Purged *k*-fold and embargo

Following AFML chapter 7, folds are *contiguous* blocks along the sample order. They are never shuffled. For a test block with observation starts \([t_0, t_{\mathrm{end}}]\), define the information interval as

\[
[t_0,\ \max(t_{\mathrm{end}},\ \max\{t_1[j] : j \in \mathrm{test}\})].
\]

A training row \(i\) with interval \([i, t_1[i]]\) is purged if the two closed intervals overlap. This is slightly more aggressive than purging only against \([t_0, t_{\mathrm{end}}]\): it also removes training rows whose labels extend into the lifetime of the last test labels. The implementation is `purge_train_indices` in `src/finmlcv/splits.py`.

Embargo is a buffer *after* each test segment. AFML specifies it as a fraction of the sample size (`embargo_pct * n`). The library also accepts a fixed bar count and a fraction of median label lifetime. Serial correlation in residuals is typically forward-looking — a shock that hits the test block still contaminates the next few training observations if those are left in the train set of a later fold — so the embargo is one-sided to the right of each test block. When CPCV places several non-adjacent groups in the same test combination, embargo is applied after *each* contiguous segment.

The sklearn contract is `split(X, y=None, groups=None) -> (train_idx, test_idx)` with integer positions into the rows of \(X\). `PurgedKFold` inherits `BaseCrossValidator`. Wrappers `NaiveKFold` and `WalkForwardSplit` exist solely so that an experiment can score an invalid protocol and a causal-but-unpurged protocol through the same loop.

## 4. Combinatorial purged CV and path reconstruction

A single purged *k*-fold evaluation produces *k* numbers. Each number is a score on one contiguous hole in the timeline. That is already more honest than shuffled *k*-fold, but it is still one backtest design: the researcher has picked a fold count and a hole placement.

CPCV (AFML ch. 12) partitions the timeline into \(N\) contiguous groups and, for every combination of \(k\) groups, holds those groups out as the test set. There are \(\binom{N}{k}\) splits. Defaults in this repository are \(N=6\), \(k=2\), hence 15 splits — a laptop budget, not a research grid. Each split is purged and embargoed against the (possibly non-adjacent) test segments.

A *path* is a complete coverage of the timeline. For each group \(g\) there are \(\binom{N-1}{k-1}\) combinations in which \(g\) is in the test set; that integer is the number of paths \(\varphi\). Path assignment is: walk the combinations in lexicographic order and, independently for each group, number the appearances of that group as 0, 1, …, \(\varphi-1\). Path \(p\) is then the concatenation, over groups \(g = 0,\ldots,N-1\), of the test predictions from the unique split in which \(g\) was held out *and* assigned path id \(p\). The unit test `test_cpcv_paths_cover_timeline` checks that every path visits every row exactly once.

Two caveats, both documented in the code.

First, a path is *not* the output of one model. Each segment is produced by a classifier trained on a different complement. The path is a device for reading a *distribution* of backtest outcomes under combinatorial holes, not a simulation of a single deployed rule.

Second, the path Sharpes are computed on the full concatenated series, whereas purged *k*-fold Sharpes in Table 3 of the README are computed per fold (short test blocks). Sampling error on a 200-bar fold Sharpe is large; sampling error on a 1,000-bar path is smaller. Comparing 2.08 (shuffled, per fold) with 1.22 (CPCV, per path) without that caveat would be a mistake. Comparing shuffled per-fold 2.08 with purged per-fold 0.16 is apples to apples, and that is the contrast the TOY panel is for.

## 5. DSR derivation sketch

Let \(r_1,\ldots,r_n\) be per-period strategy returns, \(\widehat{\mathrm{SR}}\) their sample Sharpe (not annualised), \(\hat\gamma_3\) their skewness, and \(\hat\gamma_4\) their Pearson kurtosis (3 for a Gaussian). Bailey & López de Prado (2012, 2014) approximate

\[
\widehat{\mathrm{PSR}}(\mathrm{SR}^*)
=
\Phi\!\left(
\frac{(\widehat{\mathrm{SR}}-\mathrm{SR}^*)\sqrt{n-1}}
{\sqrt{1-\hat\gamma_3\widehat{\mathrm{SR}}+\frac{\hat\gamma_4-1}{4}\widehat{\mathrm{SR}}^2}}
\right),
\]

the estimated probability that the true Sharpe exceeds a benchmark \(\mathrm{SR}^*\). The non-annualised convention is deliberate: annualisation would have to be applied to both the estimate and the benchmark, and cancels in the leading term.

The deflated Sharpe ratio is PSR evaluated at a particular benchmark: the expected *maximum* of \(N\) independent noise Sharpes. Under a Gaussian approximation of the Sharpe estimator with standard deviation \(\sigma_{\mathrm{SR}}\),

\[
\mathbb{E}[\max_{1\le j\le N}\mathrm{SR}_j]
\approx
\sigma_{\mathrm{SR}}
\left(
(1-\gamma)\,
\Phi^{-1}\!\big(1-\tfrac{1}{N}\big)
+
\gamma\,
\Phi^{-1}\!\big(1-\tfrac{1}{Ne}\big)
\right),
\]

where \(\gamma \approx 0.5772156649\) is the Euler–Mascheroni constant. The paper uses

\[
\sigma_{\mathrm{SR}}^2
=
\frac{1-\hat\gamma_3\widehat{\mathrm{SR}}+\frac{\hat\gamma_4-1}{4}\widehat{\mathrm{SR}}^2}{n-1}.
\]

For \(N=1\) the inverse-normal terms diverge; the implementation sets the haircut to 0 so that DSR reduces to \(\mathrm{PSR}(0)\).

Assumptions, all visible in `deflated_sharpe_ratio`:

- Trials are independent. Positive correlation among trials *reduces* the expected maximum; using the independent formula is then conservative (DSR too small). Negative correlation goes the other way.
- \(N\) (`n_trials`) is supplied by the caller. It is not inferred. A paper that tried 200 feature sets and reports \(N=5\) will look better than it is.
- Returns are treated as strictly stationary for the moment estimators. They are not, in markets.
- The expansion is not a Student-\(t\). For tiny \(n\) the number is a sketch.

The unit test `test_dsr_decreases_as_n_trials_increases` is a spot-check of the monotone comparative static, not a proof.

Sharpe of a constant series is defined here as 0 if the mean excess return is 0 and as NaN if the mean is non-zero with zero variance. That is an edge-case convention; it is not in Bailey–López de Prado.

## 6. The leakage auditor

`audit_leakage` returns a `LeakageReport` (dict-serialisable, printable) with three families of findings.

1. **Adjacent overlap fraction.** The share of consecutive pairs whose intervals overlap. Values near 1 are normal for every-bar sampling of multi-bar labels; they mean shuffled *k*-fold is invalid, not that the dataset is “wrong”.
2. **Future columns.** (a) A column-level or per-row `feature_times` stamp after the label start. (b) Absolute Pearson correlation of a column with \(y[t+1]\) or with \(y\) itself above a threshold (default 0.75). The second rule catches “I put the label in \(X\)” and “I put the forward return that defines the label in \(X\)”.
3. **Merge hazards.** Duplicate index on \(X\); duplicate timestamps on a series the caller admits to having merged. An inner merge on a raw stamp is not an as-of merge and can duplicate a row into a future.

The scalar `score` in \([0,1]\) is a triage heuristic. It is not a statistical test. The TOY audit of `future_column` reports `x_leak`, adjacent overlap 1.0 (horizon 30 on every-bar labels), and score 0.55.

## 7. Experimental design

Two experiments share a seed of 42 and write artefacts under `results/`.

### 7.1 Overlap-local DGP (Figure 1)

Labels are piecewise constant on 32 contiguous regimes, assigned as independent fair bits. Features are a scaled time coordinate plus \(\mathcal{N}(0, 10^{-4})\) noise. The true walk-forward AUC is 1/2: nothing measurable at \(t\) from a *causal* feature identifies the regime of a hole that has been removed from train. A *k*-NN with 11 neighbours interpolates the regime from time-neighbours. Shuffled *k*-fold leaves those neighbours in train. Purged *k*-fold and CPCV remove them.

This DGP is pedagogical. It stands in for the empirical situation in which features are highly persistent (price levels, slowly moving technicals, entity embeddings) and labels are overlapping, so that “nearby in \(X\)” means “nearby in time”.

### 7.2 Future-column DGP

The label is the sign of an independent Gaussian. That Gaussian is stored as `x_leak`. Logistic regression recovers the sign under every splitter (AUC 1.00 to reported precision). After the auditor drops the column, AUC is 0.51–0.52. The point is negative: **do not ask CPCV to save a look-ahead feature**.

### 7.3 Synthetic panel (Table 3)

A 1,200-bar geometric random walk with a weak AR(1) return, \(\phi = 0.08\), \(\sigma=0.01\). Features: lags of the return, 5- and 20-bar realised vol, 10- and 20-bar momentum, deviation from a 20-bar trailing mean. Label: sign of the 10-bar forward log return. Classifier: L2-logistic. Overlay: \(\mathrm{position}_t = +1\) if \(\hat P(y=1)\ge 1/2\), else \(-1\), times the forward log return. No costs. DSR `n_trials=20`. Embargo grid \(\{0, 0.01, 0.05\}\).

The AR coefficient is small on purpose. The experiment is about the *spread across protocols*, not about finding a strategy. Annualised Sharpes on 200-bar folds are noisy; several entries in Table 3 have Sharpe standard deviations larger than the mean. That noise is part of the result.

FULL mode, if `configs/experiment.yaml` is switched and `data/raw/` contains cached OHLCV, would repeat the same pipeline per ticker. This run did not download vendor data; Ken French / Yahoo / Stooq are attempted by `scripts/download_data.py` and skipped on failure. FI-2010 is out of scope.

## 8. TOY numbers (copied, not invented)

From `results/tables/synthetic_leakage.csv`, `overlap_local`, *k*-NN:

- Naive KFold AUC mean 0.997 (std 0.001).
- TimeSeriesSplit AUC mean 0.500.
- Purged KFold AUC mean 0.681 (std 0.167).
- CPCV AUC mean 0.456 (std 0.188).

From the same file, `future_column`, logistic: all four protocols AUC 1.000 with the leak in \(X\); 0.523 / 0.506 / 0.515 / 0.511 after the drop.

From `results/tables/benchmark_protocols.csv`, toy synthetic panel, logistic, 1,170 aligned rows:

- Naive KFold: AUC 0.561, Sharpe 2.08, DSR 0.53, fraction of folds with Sharpe > 0 equal to 1.00.
- TimeSeriesSplit: AUC 0.508, Sharpe 0.81, DSR 0.39, fraction 0.60.
- Purged KFold (embargo 1%): AUC 0.518, Sharpe 0.16, DSR 0.23, fraction 0.60.
- CPCV (5 paths): AUC 0.522, Sharpe 1.22, DSR 0.65, fraction 1.00.

Embargo 0 / 1% / 5% on purged *k*-fold: AUC 0.516 / 0.518 / 0.506. The grid does not move the answer much on this DGP.

Interpretation in one paragraph. Shuffled *k*-fold on overlapping labels with a time-local model is not an estimator of OOS skill; Figure 1 is the existence proof. On the AR(1) panel, where a *linear* model and *causal* features give a genuine but weak signal, shuffled *k*-fold still inflates the overlay Sharpe from 0.16 (purged, per fold) to 2.08. CPCV path Sharpes sit in between because they are a different functional of the predictions (full-timeline concatenations). DSR with `n_trials=20` already knocks the shuffled result from “every fold beats zero” to a probability 0.53 that the true Sharpe exceeds the expected-max noise Sharpe — i.e. the multiple-testing haircut, not the splitter, is what remains once the leak is removed, and it is not large.

## 9. Pitfalls (checklist)

**Label lifetime.** Sampling a 10-day return every day is 10-fold overlap. Sampling it every 10 days restores approximate non-overlap at the cost of sample size. Purging is the alternative to thinning.

**Embargo units.** `embargo_pct` is a fraction of *n*, not of calendar time and not of each label’s lifetime. On irregular event clocks (trades, quotes) prefer `embargo_bars` in event time.

**NaN-after-split.** Imputing or dropping inside the CV loop, using test-fold statistics, is a leak. Align first.

**Point-in-time joins.** `merge_asof` backward, not `pd.merge` on the raw stamp. Duplicate timestamps are reported by the auditor if the caller passes the joining index.

**Survivorship and look-ahead universe.** A monthly reconstitution that uses the month-end membership to score the whole month is the same family of bug as a future column.

**Costs and turnover.** A high-AUC classifier that flips every bar has overlay Sharpe that will not survive 1 bp. `turnover` is computed; it is not subtracted from the TOY overlay.

**Nested selection.** Using CPCV paths to *choose* a model and then reporting the same paths as evaluation is still selection bias. DSR’s `n_trials` is the place to confess that. A nested split is better.

**IID DSR.** If 50 random seeds of the same architecture are highly correlated, \(N=50\) over-haircuts. If 50 feature recipes were tried and one is shown, \(N=1\) under-haircuts.

**Accuracy.** Do not headline it. MCC, F1, balanced accuracy, and AUC are exported; accuracy is not.

**CPCV compute.** \(\binom{N}{k}\) grows fast. \(N=10\), \(k=3\) is already 120 fits. The TOY defaults are small.

**Path dependence of barriers.** Triple-barrier search in this library follows the observed close path between \(t\) and the vertical barrier. It does not model intra-bar excursions. Intraday first-touch would need a different price field.

## 10. Implementation notes

- Language: Python ≥ 3.11, package `finmlcv` under `src/`.
- Splitters reimplemented from the description in AFML. Not copied from mlfinlab. No GPL sources vendored.
- Deterministic seeds, default 42.
- Type hints throughout; `py.typed` shipped.
- Optional extras: `torch` (tiny CPU LSTM for the toy, not a forecasting contribution), `xgboost` (falls back to sklearn GBM).
- CI: ruff + pytest on push.
- Tests that must remain green: overlapping labels make shuffled KFold look good and purged look honest; embargo drops the post-test buffer; CPCV paths cover the timeline; Sharpe of a zero series is 0; DSR falls as `n_trials` rises; a constructed path hits the upper barrier before the lower and vice versa; a leaked future column is flagged.

## 11. What this is not

It is not a trading system. It is not a claim that purged CV “finds alpha” by shrinking Sharpe. It is not a reproduction of a published cross-section of expected returns. It is not FI-2010, not CRSP, not a point-in-time fundamentals tape. LSTM code, if PyTorch is installed, is a seed-controlled toy classifier of a few hundred parameters; it is not a contribution to sequence modelling.

The intended reader is a PhD supervisor who wants to know whether a student understands why a 0.99 AUC under `KFold` on a crypto LSTM is not a result. The intended next artefact is a paper that uses these splitters on a dataset the lab actually cares about.

## 12. Reading list

Primary methods:

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 3 (labelling), 7 (purging, embargo, *k*-fold), 12 (CPCV). [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management* 40(5), 94–107. SSRN: [2460551](https://ssrn.com/abstract=2460551)
- Bailey, D. H. and López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk* 15(2), 13–44. SSRN: [1821643](https://ssrn.com/abstract=1821643)

Backtest protocol and multiple testing:

- Arnott, R., Harvey, C. R. and Markowitz, H. (2019). A backtesting protocol in the era of machine learning. *Journal of Financial Data Science* 1(1), 64–74. [JFDS](https://www.pm-research.com/content/iijjfds/1/1/64)
- Harvey, C. R., Liu, Y. and Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies* 29(1), 5–68. [doi:10.1093/rfs/hhv059](https://doi.org/10.1093/rfs/hhv059)
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1), 13–28.
- Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting on out-of-sample performance. *Notices of the AMS* 61(5), 458–471. [PDF](https://www.ams.org/notices/201405/rnoti-p458.pdf)
- Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance* 20(4), 39–69.
- White, H. (2000). A reality check for data snooping. *Econometrica* 68(5), 1097–1126. [doi:10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152)
- Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica* 73(4), 1237–1282.

Time-series CV and ML in asset pricing:

- Bergmeir, C., Hyndman, R. J. and Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis* 120, 70–83. [doi:10.1016/j.csda.2017.11.003](https://doi.org/10.1016/j.csda.2017.11.003)
- Cerqueira, V., Torgo, L. and Mozetič, I. (2020). Evaluating time series forecasting models: an empirical study on performance estimation methods. *Machine Learning* 109, 1997–2028.
- Gu, S., Kelly, B. and Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies* 33(5), 2223–2273. [doi:10.1093/rfs/hhz033](https://doi.org/10.1093/rfs/hhz033)
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
- Dixon, M. F., Halperin, I. and Bilokon, P. (2020). *Machine Learning in Finance*. Springer.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

López de Prado (2018b), “The 10 reasons most machine learning funds fail”, *Journal of Portfolio Management*, is a short version of the methodological argument for an applied-finance audience.

## 13. Reproducibility

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make test    # 28 tests in this snapshot
make toy     # writes results/tables/*.csv and results/figures/*.png
```

The TOY CSVs and PNGs in `results/` are part of the paper artefact and are not gitignored. Vendor downloads under `data/raw/` are gitignored. No GitHub remotes are required to run this.

## Acknowledgements

The algorithms are due to López de Prado and to Bailey & López de Prado. Errors in the reimplementation are the author’s.
