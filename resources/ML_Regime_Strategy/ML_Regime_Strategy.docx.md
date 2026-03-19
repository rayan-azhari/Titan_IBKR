**ML Regime Strategy**

Production-Grade Quantitative Pipeline

vectorbt  ·  nautilus\_trader  ·  IBKR

# **Overview**

This document and the accompanying Python module implement a complete HMM-regime-detection \+ meta-labelling quantitative strategy, engineered to production standard. Every identified failure mode of the initial conceptual blueprint has been resolved with working code.

| Core Invariant:  The single most important design decision: one FeatureEngine class, called identically in research (batch) and in Nautilus (step). If features are not byte-for-byte identical between training and production, every other fix is irrelevant. |
| :---- |

## **Resolved Issues**

| Issue | Fix Applied |
| :---- | :---- |
| **HMM single-step inference bug** | predict\_current\_state() runs forward-backward on full rolling window; takes last timestep posterior |
| **HMM label switching** | Canonical ordering by mean volatility ascending, deterministic across retraining cycles |
| **Frac-diff window mismatch** | tau-truncated window computed once at init; same deque size used in research and production |
| **Exit logic gap** | compute\_barriers\_for\_live\_order() mirrors label() with identical parameters and regime-conditional multipliers |
| **Data leakage in CV** | PurgedKFold: purges training samples whose label spans overlap test window; embargoes post-test buffer |
| **Hardcoded 0.6 threshold** | Threshold tuned on OOF predictions via grid search; stored with model; configurable target metric |
| **Position sizing stub** | FractionalKelly: quarter-Kelly, vol-targeting cap, hard max-position cap, all three applied in order |
| **No model lifecycle plan** | ModelHealthMonitor (KL drift, prob drift, calibration); RetrainingScheduler (scheduled \+ alert-triggered) |
| **Optimistic transaction costs** | vectorbt backtest includes slippage parameter; 10bps fees \+ 10bps slippage as baseline |
| **Sample weight ignored** | compute\_sample\_uniqueness() weights training samples; overlapping labels downweighted |

# **Module Structure**

The strategy is implemented as a Python package with one responsibility per module. The dependency graph is strictly acyclic: research\_pipeline and nautilus\_strategy both depend on the lower modules; the lower modules have no cross-dependencies.

| File | Used in | Responsibility |
| :---- | :---- | :---- |
| feature\_engine.py | Both | FeatureEngine: step() for production, batch() wraps step() for research |
| regime\_detection.py | Both | RegimeDetector: HMM with canonical ordering and proper sequence inference |
| labelling.py | Research \+ Production | TripleBarrierLabeller: generates labels in research; replicates barriers live |
| cross\_validation.py | Research | PurgedKFold \+ sample uniqueness weights |
| meta\_labeller.py | Both | MetaLabeller: XGBoost with OOF threshold tuning |
| position\_sizing.py | Both | FractionalKelly \+ WinLossTracker |
| model\_lifecycle.py | Production | ModelHealthMonitor \+ RetrainingScheduler |
| nautilus\_strategy.py | Production | Nautilus Strategy subclass wiring all components |
| research\_pipeline.py | Research | run() orchestrates full research workflow |

# **Phase 1: Feature Engineering**

## **FeatureEngine — The Research/Production Parity Guarantee**

The FeatureEngine computes three stationary features: log\_return, ewm\_vol (exponentially weighted volatility), and frac\_diff (fractionally differenced price). The critical design is that batch() calls step() sequentially — it is not a separate batch implementation. This is the architectural guarantee.

### **Fractional Differentiation Window**

The frac-diff weights decay according to the binomial series expansion of (1-L)^d. The tau threshold (default 1e-4) truncates this infinite series to a finite window. That window is computed once at init and becomes the maxlen of the price deque. Both research and production use this identical window.

fe \= FeatureEngine(vol\_span=20, frac\_d=0.4, frac\_tau=1e-4)

\# In research:

features \= fe.batch(prices)           \# returns DataFrame

\# In Nautilus on\_bar():

feat \= self.feature\_engine.step(close)  \# returns np.ndarray or None

| Parameter Note:  Choosing frac\_d: d=0 is the raw price (non-stationary). d=1 is simple returns (maximum differentiation, loses all memory). d=0.4 is a common choice that achieves near-stationarity while retaining significant long-memory. Validate by testing ADF stationarity on your specific instrument. |
| :---- |

# **Phase 2: Regime Detection**

## **RegimeDetector — Two Bugs, Two Fixes**

### **Bug 1: Single-Step Inference**

Gemini's predict(single\_observation) discards the transition probability context. The forward-backward algorithm — which gives HMMs their predictive power — requires a sequence. The fix: maintain a rolling window of min\_seq\_len observations and call predict\_proba() on the full window, then take the last timestep's posterior.

\# WRONG (Gemini):

state \= model.predict(\[current\_features\])

\# CORRECT:

current\_state, posteriors \= detector.predict\_current\_state(feature\_window)

\# feature\_window: last min\_seq\_len observations, most recent last

### **Bug 2: Label Switching**

Unsupervised models have no ordinal stability. After retraining, what was 'Regime 0' may become 'Regime 1'. Any hardcoded regime-conditional logic silently inverts. The fix: sort states by their mean volatility feature after fitting. State 0 is always the lowest-volatility state, State 1 the highest. This ordering is physically interpretable and reproducible.

\# At fit time:

raw\_vol\_means \= model.means\_\[:, vol\_feature\_idx\]

self.\_canonical\_to\_raw \= np.argsort(raw\_vol\_means)  \# stored, not recomputed

\# At inference:

canonical\_posteriors \= raw\_posteriors\[-1\]\[self.\_canonical\_to\_raw\]

| Empirical Validation Required:  Validate min\_seq\_len empirically. The posterior quality for the last timestep degrades when the sequence is short relative to the HMM's learned transition timescales. Run posterior confidence as a function of sequence length on your training data; find where it plateaus. |
| :---- |

# **Phase 3: Triple Barrier Labelling**

## **The Exit Logic Gap — Resolved**

Gemini generated triple barrier labels in research but left exits undefined in production (a comment: 'Define your exit logic'). The consequence: XGBoost was trained on outcomes produced by one exit regime and deployed into a different one. The model cannot generalise across this misalignment.

The fix: TripleBarrierLabeller exposes two methods that share identical parameter logic:

* **Research:** label() generates training labels. Called once in research.

* **Production:** compute\_barriers\_for\_live\_order() computes take-profit and stop-loss price levels for a live order. Called in nautilus\_strategy.on\_bar() at order submission.

Both methods apply the same regime-conditional multipliers:

\# Regime 1 (high vol / trending): wider take-profit, tighter stop-loss

REGIME\_UPPER\_SCALE \= {0: 1.0, 1: 1.5}

REGIME\_LOWER\_SCALE \= {0: 1.0, 1: 0.7}

The live strategy then manages the position bar-by-bar in \_manage\_position(), checking close price against the stored barrier levels on every bar until one is hit or max\_holding\_bars is reached. This is the Triple Barrier state machine.

# **Phase 4: Purged and Embargoed Cross-Validation**

## **Why Standard K-Fold Fails**

Standard K-Fold has two leakage mechanisms that are especially severe for triple-barrier labels:

* **1\. Correlation:** Serial correlation: an observation at t and t+1 share macroeconomic context. A model trained on t effectively sees the context of t+1.

* **2\. Label span:** Label look-ahead: a triple barrier label spans \[entry\_time, exit\_time\]. If that span extends into the test window, the training observation contains forward-looking information about test-period prices.

### **Purging**

Removes training observations whose label span overlaps with the test window. Overlap condition: entry\_time \<= test\_end AND exit\_time \>= test\_start. This directly removes the label look-ahead leak.

### **Embargoing**

Removes a contiguous block of observations immediately after each test fold. Default: 1% of total observations. This accounts for residual serial correlation that persists after the test window.

### **Sample Uniqueness Weights**

When multiple concurrent labels share the same bars, each observation contributes less independent information. compute\_sample\_uniqueness() measures the fraction of each label's bar-span that is unique to it. Observations with low uniqueness receive lower sample weight in XGBoost training.

uniqueness \= compute\_sample\_uniqueness(entry\_times, exit\_times, bar\_times)

\# Returns Series: mean uniqueness per label, range (0, 1\]

\# Used as sample\_weight in model.fit()

# **Phase 5: Meta-Labeller**

## **Threshold Tuning — OOF, Not Manual**

Gemini's 0.6 threshold was chosen by inspection. This is overfitting-by-hand. The threshold needs to be treated as a hyperparameter, tuned on out-of-fold predictions from the same purged CV used for training, and stored with the model so production uses the identical value.

The MetaLabeller fits with PurgedKFold, collects OOF probability scores across all folds, then grid-searches the threshold on those OOF scores. The target metric (F1, precision, or recall) is configurable depending on whether you want to maximise hit rate or trade count.

ml \= MetaLabeller(threshold\_target='f1')

ml.fit(X, meta\_y, exit\_times, bar\_times=prices.index)

\# CV results include:

\# mean\_auc, oof\_f1, oof\_precision, oof\_recall, optimal\_threshold

| Threshold Target Choice:  The threshold matters directionally: lower threshold \= more trades, lower precision; higher \= fewer trades, higher precision. F1 balances both. If you are capital-constrained and want high-confidence trades, tune for precision. If you want maximum throughput, tune for recall. |
| :---- |

# **Phase 6: Position Sizing**

## **Fractional Kelly with Dual Caps**

The meta-labeller's probability output is more useful as a continuous signal than a binary filter. Quarter-Kelly maps it to a portfolio fraction. Two hard constraints prevent aggressive Kelly behaviour from becoming dangerous in practice:

* **Cap 1:** Volatility targeting: position size is capped so that a single trade contributes no more than vol\_target\_pct (default 1%) to daily portfolio volatility.

* **Cap 2:** Hard maximum: position is capped at max\_position\_pct (default 5%) of portfolio regardless of Kelly or vol output.

The effective position is the minimum of Kelly-implied size, vol-targeting cap, and hard cap. For most typical conditions, the vol cap binds first.

sizer \= FractionalKelly(fraction=0.25, max\_position\_pct=0.05, vol\_target\_pct=0.01)

size \= sizer.size(prob\_success, wl\_tracker.ratio, nav, daily\_vol)

| Win/Loss Ratio Prior:  The win/loss ratio is estimated from live trade history via WinLossTracker. The prior (before 10 wins and 10 losses are observed) defaults to 1.5 — a conservative assumption. Do not use backtest win/loss ratios as the prior: they overfit. |
| :---- |

# **Phase 7: Model Lifecycle Management**

## **Degradation Monitoring**

A model trained in one market environment will degrade as conditions change. ModelHealthMonitor tracks three degradation signals and fires configurable alerts:

* **1\.** Regime drift: KL divergence between live regime frequency distribution and training-time distribution. Alert when KL \> 0.2 (configurable).

* **2\.** Probability distribution drift: z-score of live mean XGBoost probability vs training OOF mean. Alert when |z| \> 2.0.

* **3\.** Calibration drift: absolute deviation between expected hit rate (mean predicted prob) and realised hit rate over last N trades. Alert when deviation \> 15%.

## **Retraining Scheduler**

Retraining is triggered by two conditions: scheduled (every 252 bars, approximately annually on daily data) or alert-triggered. Either condition sets a retrain\_requested flag that the strategy checks. After retraining, the scheduler resets and enters a new warm-up period — the new model must accumulate min\_history bars before producing features and min\_seq\_len bars before HMM inference.

| Operational Rule:  Never retrain silently in production. Every retraining event should be logged with timestamp, trigger reason, and key metrics before and after. The model that goes live must be validated on a hold-out period before activation. |
| :---- |

# **Phase 8: Production Execution (Nautilus Trader)**

## **Bar Processing Flow**

The on\_bar() handler follows a strict sequential gate logic. Each gate must pass before the next is entered:

| \# | Gate | Logic |
| :---- | :---- | :---- |
| **1** | **Feature warm-up** | feature\_engine.step() returns None until frac\_window \+ vol\_span bars accumulated |
| **2** | **HMM warm-up** | feature\_window deque must contain \>= hmm\_min\_seq\_len observations |
| **3** | **Retrain warm-up** | RetrainingScheduler.in\_warmup() must be False |
| **4** | **Position management** | If already long: check barriers every bar, exit if hit or time elapsed. Return early. |
| **5** | **Primary signal** | \_primary\_signal(): MA crossover, fast crosses above slow |
| **6** | **Meta-label filter** | meta\_labeller.predict\_proba() on current features \+ regime posteriors |
| **7** | **Kelly sizing** | FractionalKelly.size(): returns 0 if below min\_edge or Kelly is negative |
| **8** | **Barrier computation** | compute\_barriers\_for\_live\_order() called before order submission, stored in self.\_barriers |
| **9** | **Order submission** | market order, quantity \= size / close |

# **Research Pipeline Usage**

research\_pipeline.run() orchestrates all research phases and produces trained models plus the metadata required to initialise the Nautilus strategy config:

from strategy.research\_pipeline import run

results \= run(

    prices=your\_price\_series,      \# pd.Series with DatetimeIndex

    output\_dir='models',

    vol\_span=20, frac\_d=0.4,       \# must match Nautilus config

    vol\_mult\_upper=2.0,            \# must match Nautilus config

    max\_holding\_bars=5,            \# must match Nautilus config

)

meta \= results\['training\_metadata'\]

\# Use meta\['training\_regime\_dist'\], meta\['training\_prob\_mean'\],

\# meta\['training\_prob\_std'\] directly in MLRegimeStrategyConfig

The pipeline prints a full diagnostic at each phase: feature warm-up requirements, regime distributions with state descriptions, label balance, CV purge counts per fold, OOF metrics, top feature importances, and backtest summary.

# **Known Limitations and Further Work**

* **Primary signal:** The primary signal (MA crossover) is deliberately simple — it is the source of raw entry candidates, not the alpha. The meta-labeller carries the predictive weight. Replacing the primary signal with a richer model (e.g. momentum Z-score, statistical arbitrage spread) is the most direct path to higher signal quality.

* **HMM sequence length:** The HMM posterior quality degrades for sequences shorter than the model's effective transition timescale. Validate min\_seq\_len empirically on your training data before deployment.

* **Frac-diff d parameter:** fractional differentiation stationarity should be verified per instrument. The frac\_d=0.4 default is a reasonable starting point; run ADF tests at different d values and choose the minimum d achieving stationarity.

* **Kelly assumptions:** Kelly sizing assumes serial independence between trades. In practice, trades may be correlated (e.g. during trending regimes). A conservative fraction=0.25 partially mitigates this but does not eliminate it.

* **Frequency scaling:** The backtest uses daily bars. For intraday application, vol\_target\_pct, barrier multipliers, and max\_holding\_bars all need recalibration to the appropriate timescale.

* **Transaction costs:** IBKR fee structure is per-share for US equities, not a flat percentage. Replace fees=0.001 with an instrument-specific model before drawing conclusions from backtest results.

ML Regime Strategy  |  All code in accompanying /strategy/ package  |  