## Conclusions

### 1. Window Labeling Threshold is Critical

The v1 notebook used a >50% majority-vote rule to label windows as "attack." Because
UNSW-NB15 contains only ~3.5% attack flows per-file (e.g. 35/1000 in the first 1K rows),
attacks almost never dominate a 30 s window — resulting in just 21/2253 (0.93%) attack
windows in training. At that ratio any classifier learns "always predict Normal."

Lowering the threshold to >5% produced **751 attack windows out of 2253 (33.3%)** in training,
making the problem tractable. The exact percentage can be tuned per deployment scenario;
5% is a reasonable starting point given the dataset's attack density.

### 2. Class-Balanced Training Recovered Recall

With the corrected labels, adding `class_weight='balanced'` to RandomForest boosted
**attack recall from 0.18 (v1) to 0.91 (v2)** — a 5x improvement. Without balancing,
the 2:1 Normal-to-Attack ratio in training still biased the classifier toward Normal.
The balanced RF achieved **F1 = 0.90** on test data (P = 0.90, R = 0.91).

### 3. GradientBoosting Matched RF, With Different Feature Focus

GB achieved a comparable **F1 = 0.90** (P = 0.92, R = 0.84 at tuned threshold) but
concentrated 54% of feature importance on a single feature (`f0_dstip`), whereas RF
spread importance more evenly (top feature at 24%). This suggests GB found a strong
but narrow signal — distinct destination IPs per window — while RF uses a broader
combination. Both ROC AUCs are close: RF = 0.880, GB = 0.872.

### 4. Probability Threshold Tuning: Trade-off, Not Free Lunch

Tuning the decision threshold on the training PR curve (RF optimal at 0.602, GB at 0.753)
increased Normal recall from 0.55 to 0.68 (RF) and 0.61 to 0.69 (GB), at the cost of
lower attack recall (0.91 → 0.86 and 0.90 → 0.84). Overall F1 dropped slightly
(0.90 → 0.89), because the train/test class distributions differ (33% vs 81% attack).
In a security setting where **missing attacks is costlier than false alarms**, the
default balanced RF (recall = 0.91) is preferable.

### 5. Window Size Comparison (10s / 30s / 60s)

| Window | Train Windows | Test Windows | Attack % (test) | RF F1 |
|--------|--------------|-------------|-----------------|-------|
| 10 s   | 6,593        | 2,162       | 77.7%           | 0.896 |
| 30 s   | 2,253        | 734         | 81.5%           | 0.904 |
| 60 s   | 1,119        | 370         | 85.1%           | 0.907 |

All three window sizes achieve F1 > 0.89. Larger windows produce slightly higher F1
because more events per window stabilize sketch estimates, but at the cost of **coarser
temporal resolution** — a 60 s window merges distinct attack/normal phases. The 30 s
window offers a good balance: enough events for reliable sketches while retaining
sub-minute granularity for incident response.

### 6. Sketch vs Exact Feature Accuracy

| Feature group | Mean Rel Error | Correlation |
|---------------|---------------|-------------|
| CMS / CS point queries | 0.000 | 1.000 |
| Byte ratio (exact accum.) | 0.000 | 1.000 |
| F1 (Morris counter) | 0.105 | 0.998 |
| F2 (AMS estimator) | 0.22–0.36 | 0.990–0.994 |
| FM (F0 estimator) | 1.29–1.69 | 0.48–0.79 |

The **FM/F0 estimator has the highest error** (~130–170% mean relative error) due to the
leading-zeros method's inherent power-of-2 granularity. Despite this, the RF trained on
sketch features still achieves F1 = 0.90 vs the exact baseline's 0.92, because the
classifier compensates via the more accurate F1/F2/CMS features.

### 7. CMS Width Has No Effect on Detection Quality

Varying CMS width from 512 to 8,192 columns (98 KB → 338 KB per window) produced
**identical F1 = 0.90 across all configurations**. This means the CMS-derived features
(`cms_max_srcip`, `cms_topk_frac`) are already accurate enough at width = 512. The
classifier's performance is bounded by the FM estimator's noise, not CMS precision.
For this workload, the smallest CMS (4 x 512 = 16 KB) is sufficient.

### 8. Derived Features Add Value Without New Sketches

Three derived features — port-scan ratio (`f0_dstport_per_srcip`), src/dst IP ratio,
and byte asymmetry — were computed from existing sketch outputs and simple byte
accumulators (O(1) extra memory). `srcip_to_dstip_ratio` ranked 4th in RF importance
(0.092), confirming that combining sketch outputs multiplicatively can capture patterns
(e.g., many-to-one DDoS traffic) that individual sketches miss.

### 9. Rule-Based Detector Fails on Shifted Test Distribution

The rule-based baseline achieved 100% attack recall but 0% Normal recall (precision = 0.81),
effectively predicting "Attack" for every window. This reflects the test set's 81.5% attack
rate: the rules, tuned on training data (33% attack), fire too aggressively on the denser
attack traffic in the test set. ML classifiers are more robust to this distribution shift.

### 10. Practical Takeaways

- **146 KB of sketch memory** per 30 s window (CMS 4x2048 config) is sufficient to achieve
  F1 = 0.90 — only 2.9 percentage points below the exact baseline (F1 = 0.92) which
  requires unbounded memory.
- Throughput on CPU (NumPy) was **~33,000 events/sec**, processing all 2.54M records in ~76 s.
- The largest accuracy gap is in FM (F0) estimation. Switching to a min-hash or
  HyperLogLog variant could close the sketch-vs-exact gap further.
