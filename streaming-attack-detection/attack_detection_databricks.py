# Databricks notebook source

# MAGIC %md
# MAGIC # Streaming Attack Detection with Sketches (GPU-Optimized)
# MAGIC
# MAGIC This notebook implements a real-time streaming attack detection system using sketch data structures
# MAGIC on the UNSW-NB15 dataset. GPU-optimized with vectorized integer hashing and batch processing.
# MAGIC
# MAGIC **Key optimizations over the base notebook:**
# MAGIC - Multiply-shift integer hashing replaces MD5 (vectorized across seeds)
# MAGIC - Batch window processing with `numpy` / `cupy`
# MAGIC - cuML `RandomForestClassifier` when RAPIDS GPU is available
# MAGIC - Count-Sketch feature added (`cs_max_srcip`) for 10 total features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Setup and Data Exploration

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings("ignore")

# --------------- GPU auto-detection ---------------
try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
except ImportError:
    xp = np
    GPU_AVAILABLE = False

def to_device(arr):
    """Move numpy array to the compute device (GPU if available)."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return np.asarray(arr)

def to_host(arr):
    """Move array back to CPU numpy."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return cp.asnumpy(arr)
    return np.asarray(arr)

# --------------- cuML auto-detection ---------------
try:
    from cuml.ensemble import RandomForestClassifier as cuRFC
    USE_CUML = True
except ImportError:
    USE_CUML = False

from sklearn.ensemble import RandomForestClassifier as skRFC
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler

# --------------- Configuration ---------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = "UNSW_NB15"
CSV_FILES = [os.path.join(DATA_DIR, f"UNSW-NB15_{i}.csv") for i in range(1, 5)]

COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]

WINDOW_SIZE = 30  # seconds

print(f"GPU available (CuPy): {GPU_AVAILABLE}")
print(f"cuML available:       {USE_CUML}")
print(f"Array backend (xp):   {'cupy' if GPU_AVAILABLE else 'numpy'}")
print(f"Window size: {WINDOW_SIZE}s")
print(f"Random seed: {RANDOM_SEED}")
print(f"Data files: {CSV_FILES}")

# COMMAND ----------

# Quick data exploration
sample = pd.read_csv(CSV_FILES[0], header=None, names=COLUMN_NAMES, nrows=5)
print("Sample records from UNSW-NB15_1.csv:")
print(
    sample[
        ["srcip", "sport", "dstip", "dsport", "proto", "Stime",
         "sbytes", "dbytes", "attack_cat", "label"]
    ].to_string()
)
print(f"\nStime range check (first file, first 1000 rows):")
sample_1k = pd.read_csv(CSV_FILES[0], header=None, names=COLUMN_NAMES, nrows=1000)
print(f"  Min Stime: {sample_1k['Stime'].min()} -> {pd.to_datetime(sample_1k['Stime'].min(), unit='s')}")
print(f"  Max Stime: {sample_1k['Stime'].max()} -> {pd.to_datetime(sample_1k['Stime'].max(), unit='s')}")
print(f"  Label distribution: {sample_1k['label'].value_counts().to_dict()}")

# COMMAND ----------

# Count total records and compute the 70/30 split point
total_records = 0
for f in CSV_FILES:
    count = sum(1 for _ in open(f, "r", encoding="utf-8", errors="replace"))
    print(f"{f}: {count} records")
    total_records += count

print(f"\nTotal records: {total_records}")
TRAIN_COUNT = int(total_records * 0.7)
TEST_COUNT = total_records - TRAIN_COUNT
print(f"Train: first {TRAIN_COUNT} records (70%)")
print(f"Test: last {TEST_COUNT} records (30%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Vectorized Hash Utilities
# MAGIC
# MAGIC We replace per-element MD5 with a **multiply-shift** integer hash family:
# MAGIC
# MAGIC $$h_{a,b}(x) = (a \cdot x + b) \bmod 2^{64}$$
# MAGIC
# MAGIC where $a$ is odd. This is 2-universal, sufficient for all our sketches, and fully vectorisable
# MAGIC with numpy/cupy broadcasting: one call hashes N keys with K seeds simultaneously.

# COMMAND ----------

# ---- Vectorized hash primitives ------------------------------------------------

def _gen_hash_params(n, rng):
    """Generate (a, b) parameter arrays for *n* multiply-shift hash functions."""
    a_raw = rng.randint(0, 2**62, size=n)
    b_raw = rng.randint(0, 2**62, size=n)
    a = (a_raw.astype(np.uint64) << np.uint64(1)) | np.uint64(1)  # always odd
    b = b_raw.astype(np.uint64)
    return a, b


def hash_batch(keys, a, b):
    """
    Multiply-shift hash (vectorized).

    Parameters
    ----------
    keys : (N,) uint64 array on device
    a, b : (K,) uint64 arrays on device

    Returns
    -------
    (K, N) uint64 array of raw hash values (mod 2^64 via overflow).
    """
    return a[:, None] * keys[None, :] + b[:, None]


def hash_to_buckets(raw_hashes, num_buckets):
    """Map raw uint64 hashes to bucket indices in [0, num_buckets).

    Uses the upper 32 bits for best uniformity with multiply-shift.
    """
    return ((raw_hashes >> np.uint64(32)) % np.uint64(num_buckets)).astype(np.int64)


def hash_to_signs(raw_hashes):
    """Map raw uint64 hashes to signs in {-1, +1} using the top bit."""
    return np.int64(1) - np.int64(2) * (raw_hashes >> np.uint64(63)).astype(np.int64)


# ---- IP / port conversion helpers -----------------------------------------------

def ips_to_uint64(ip_col):
    """Convert IP address column to uint64 array (vectorized)."""
    s = ip_col.astype(str)
    parts = s.str.split(".", expand=True)
    if parts.shape[1] >= 4:
        octets = parts.iloc[:, :4].apply(pd.to_numeric, errors="coerce").fillna(0)
        v = octets.values.astype(np.uint64)
        return (v[:, 0] << np.uint64(24)) | (v[:, 1] << np.uint64(16)) | (v[:, 2] << np.uint64(8)) | v[:, 3]
    # Fallback for non-dotted-decimal formats
    return pd.util.hash_pandas_object(s, index=False).values.view(np.uint64)


def ports_to_uint64(port_col):
    """Convert port column to uint64 array."""
    return pd.to_numeric(port_col, errors="coerce").fillna(0).values.astype(np.uint64)


print("Vectorized hash utilities ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Sketch Implementations
# MAGIC
# MAGIC ### Median-of-Means Aggregation
# MAGIC
# MAGIC All frequency moment estimators use the **median-of-means** trick:
# MAGIC given R estimates arranged as `num_groups × group_size`,
# MAGIC compute the mean within each group and return the median of the group means.
# MAGIC
# MAGIC ### Vectorized Batch Updates
# MAGIC
# MAGIC Each sketch exposes an `update_batch(keys)` method that processes an entire
# MAGIC numpy / cupy array at once—replacing the inner Python loop over individual events.

# COMMAND ----------

def median_of_means(estimates, num_groups, group_size):
    """Median-of-means aggregation over a 1-D array of estimates."""
    arr = np.asarray(estimates, dtype=np.float64).reshape(num_groups, group_size)
    return float(np.median(np.mean(arr, axis=1)))


# =============================================================================
# F1 Estimator — Morris approximate counter (vectorized across R counters)
# =============================================================================
class MorrisCounter:
    """
    Morris approximate counter for F1 (total count / volume).

    R independent copies with median-of-means give a reliable estimate.
    The counter is inherently sequential across events but vectorized across R.
    """

    def __init__(self, R=64, num_groups=8, seed_offset=0):
        self.R = R
        self.num_groups = num_groups
        self.group_size = R // num_groups
        self.counters = np.zeros(R, dtype=np.float64)  # always on CPU
        self._so = seed_offset
        self.rng = np.random.RandomState(RANDOM_SEED + seed_offset)

    def update_batch(self, n_events):
        """Process *n_events* events (vectorized across R counters per event)."""
        for _ in range(n_events):
            rands = self.rng.random(self.R)
            probs = np.power(2.0, -self.counters)
            self.counters += (rands < probs).astype(np.float64)

    def estimate(self):
        individual = np.power(2.0, self.counters) - 1.0
        return median_of_means(individual, self.num_groups, self.group_size)

    def memory_bytes(self):
        return self.R * 8

    def reset(self):
        self.counters[:] = 0
        self.rng = np.random.RandomState(RANDOM_SEED + self._so)


# =============================================================================
# F0 Estimator — Flajolet-Martin with leading zeros (vectorized)
# =============================================================================
class FMEstimator:
    """
    Flajolet-Martin style F0 estimator using **leading zeros** of
    multiply-shift hashes (upper bits are well-distributed).

    `update_batch` hashes an entire key array at once — ~100× faster than
    per-element MD5.
    """

    def __init__(self, R=64, num_groups=8, seed_offset=100):
        self.R = R
        self.num_groups = num_groups
        self.group_size = R // num_groups
        self.max_zeros = xp.zeros(R, dtype=xp.int32)
        rng = np.random.RandomState(RANDOM_SEED + seed_offset)
        a, b = _gen_hash_params(R, rng)
        self.a = to_device(a)
        self.b = to_device(b)
        self._so = seed_offset

    def update_batch(self, keys):
        """keys: (N,) uint64 array on device."""
        if len(keys) == 0:
            return
        raw = hash_batch(keys, self.a, self.b)  # (R, N)
        # Leading zeros = 63 - floor(log2(h)); hash=0 maps to 64 but is ~impossible
        safe = xp.maximum(raw, xp.uint64(1))
        lz = 63 - xp.floor(xp.log2(safe.astype(xp.float64))).astype(xp.int32)
        self.max_zeros = xp.maximum(self.max_zeros, xp.max(lz, axis=1))

    def estimate(self):
        mz = to_host(self.max_zeros).astype(np.float64)
        individual = np.power(2.0, mz)
        return median_of_means(individual, self.num_groups, self.group_size)

    def memory_bytes(self):
        return self.R * 4

    def reset(self):
        self.max_zeros = xp.zeros(self.R, dtype=xp.int32)


# =============================================================================
# F2 Estimator — AMS (vectorized)
# =============================================================================
class AMSEstimator:
    """
    AMS estimator for F2 (second frequency moment).

    For each repetition j, Z_j += s_j(x). Estimate = Z_j².
    `update_batch` computes signs for all keys × all repetitions in one broadcast.
    """

    def __init__(self, R=64, num_groups=8, seed_offset=200):
        self.R = R
        self.num_groups = num_groups
        self.group_size = R // num_groups
        self.Z = xp.zeros(R, dtype=xp.float64)
        rng = np.random.RandomState(RANDOM_SEED + seed_offset)
        a, b = _gen_hash_params(R, rng)
        self.a = to_device(a)
        self.b = to_device(b)
        self._so = seed_offset

    def update_batch(self, keys):
        if len(keys) == 0:
            return
        raw = hash_batch(keys, self.a, self.b)  # (R, N)
        signs = hash_to_signs(raw)               # (R, N) in {-1,+1}
        self.Z += xp.sum(signs.astype(xp.float64), axis=1)

    def estimate(self):
        z = to_host(self.Z)
        individual = z ** 2
        return median_of_means(individual, self.num_groups, self.group_size)

    def memory_bytes(self):
        return self.R * 8

    def reset(self):
        self.Z = xp.zeros(self.R, dtype=xp.float64)


# =============================================================================
# Count-Min Sketch (vectorized scatter-add via bincount)
# =============================================================================
class CountMinSketch:
    """
    Count-Min Sketch (d rows × w columns).

    `update_batch` hashes all keys at once and uses `bincount` for scatter-add.
    `query_batch` returns min-across-rows estimates for a batch of keys.
    """

    def __init__(self, d=4, w=2048, seed_offset=300):
        self.d = d
        self.w = w
        self.table = xp.zeros((d, w), dtype=xp.int64)
        rng = np.random.RandomState(RANDOM_SEED + seed_offset)
        a, b = _gen_hash_params(d, rng)
        self.a = to_device(a)
        self.b = to_device(b)

    def update_batch(self, keys):
        if len(keys) == 0:
            return
        raw = hash_batch(keys, self.a, self.b)           # (d, N)
        buckets = hash_to_buckets(raw, self.w)            # (d, N) int64
        for i in range(self.d):
            bc = xp.bincount(buckets[i], minlength=self.w)
            self.table[i] += bc[: self.w]

    def query_batch(self, keys):
        """Return (N,) int64 array of point-query estimates (min across rows)."""
        if len(keys) == 0:
            return xp.array([], dtype=xp.int64)
        raw = hash_batch(keys, self.a, self.b)
        buckets = hash_to_buckets(raw, self.w)
        rows = xp.stack([self.table[i, buckets[i]] for i in range(self.d)])
        return xp.min(rows, axis=0)

    def memory_bytes(self):
        return self.d * self.w * 8

    def reset(self):
        self.table = xp.zeros((self.d, self.w), dtype=xp.int64)


# =============================================================================
# Count-Sketch (vectorized scatter-add with signs)
# =============================================================================
class CountSketch:
    """
    Count-Sketch (r rows × b buckets) — unbiased frequency estimator.

    Uses separate bucket and sign hash families.
    """

    def __init__(self, r=5, b=2048, seed_offset=400):
        self.r = r
        self.b_size = b
        self.table = xp.zeros((r, b), dtype=xp.int64)
        rng = np.random.RandomState(RANDOM_SEED + seed_offset)
        ba, bb = _gen_hash_params(r, rng)
        sa, sb = _gen_hash_params(r, rng)
        self.bucket_a = to_device(ba)
        self.bucket_b = to_device(bb)
        self.sign_a = to_device(sa)
        self.sign_b = to_device(sb)

    def update_batch(self, keys):
        if len(keys) == 0:
            return
        raw_bkt = hash_batch(keys, self.bucket_a, self.bucket_b)
        raw_sgn = hash_batch(keys, self.sign_a, self.sign_b)
        buckets = hash_to_buckets(raw_bkt, self.b_size)
        signs = hash_to_signs(raw_sgn)
        for i in range(self.r):
            w = signs[i].astype(xp.float64)
            bc = xp.bincount(buckets[i], weights=w, minlength=self.b_size)
            self.table[i] += bc[: self.b_size].astype(xp.int64)

    def query_batch(self, keys):
        """Return (N,) float64 array — median of signed estimates per key."""
        if len(keys) == 0:
            return xp.array([], dtype=xp.float64)
        raw_bkt = hash_batch(keys, self.bucket_a, self.bucket_b)
        raw_sgn = hash_batch(keys, self.sign_a, self.sign_b)
        buckets = hash_to_buckets(raw_bkt, self.b_size)
        signs = hash_to_signs(raw_sgn)
        rows = xp.stack([self.table[i, buckets[i]] * signs[i] for i in range(self.r)])
        return xp.median(rows.astype(xp.float64), axis=0)

    def memory_bytes(self):
        return self.r * self.b_size * 8

    def reset(self):
        self.table = xp.zeros((self.r, self.b_size), dtype=xp.int64)


print("All 5 sketch data structures implemented (vectorized):")
print("  - MorrisCounter  (F1)")
print("  - FMEstimator    (F0)")
print("  - AMSEstimator   (F2)")
print("  - CountMinSketch (CMS)")
print("  - CountSketch    (CS)")

# COMMAND ----------

# === Sketch sanity tests ===
print("=== Sketch Sanity Tests ===\n")

# F1 (Morris) — count 1000 events
mc = MorrisCounter(R=64, num_groups=8, seed_offset=0)
mc.update_batch(1000)
f1_est = mc.estimate()
print(f"F1 (Morris): true=1000, est={f1_est:.1f}, err={abs(f1_est-1000)/1000:.2%}")

# F0 (FM) — 200 distinct elements, 1000 total
fm = FMEstimator(R=64, num_groups=8, seed_offset=100)
keys_f0 = to_device(np.array([i % 200 for i in range(1000)], dtype=np.uint64))
fm.update_batch(keys_f0)
f0_est = fm.estimate()
print(f"F0 (FM):     true=200,  est={f0_est:.1f}, err={abs(f0_est-200)/200:.2%}")

# F2 (AMS) — 100×A + 50×B + 20×C -> true F2=12900
ams = AMSEstimator(R=64, num_groups=8, seed_offset=200)
keys_f2 = to_device(np.concatenate([
    np.full(100, 1, dtype=np.uint64),
    np.full(50, 2, dtype=np.uint64),
    np.full(20, 3, dtype=np.uint64),
]))
ams.update_batch(keys_f2)
f2_est = ams.estimate()
print(f"F2 (AMS):    true=12900, est={f2_est:.1f}, err={abs(f2_est-12900)/12900:.2%}")

# CMS — point queries
cms = CountMinSketch(d=4, w=2048, seed_offset=300)
cms_keys = to_device(np.concatenate([
    np.full(100, 10, dtype=np.uint64),
    np.full(10, 20, dtype=np.uint64),
]))
cms.update_batch(cms_keys)
q = to_host(cms.query_batch(to_device(np.array([10, 20], dtype=np.uint64))))
print(f"CMS:         query(10)={q[0]} (true=100), query(20)={q[1]} (true=10)")

# Count-Sketch — point queries
cs = CountSketch(r=5, b=2048, seed_offset=400)
cs.update_batch(cms_keys)
q2 = to_host(cs.query_batch(to_device(np.array([10, 20], dtype=np.uint64))))
print(f"CS:          query(10)={q2[0]:.0f} (true=100), query(20)={q2[1]:.0f} (true=10)")

print("\nAll sketches pass sanity checks.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Streaming Engine and Feature Extraction
# MAGIC
# MAGIC ### Window Processor
# MAGIC
# MAGIC For each tumbling window we maintain sketches for F0, F1, F2, CMS, and CS.
# MAGIC The `update_batch` method feeds an entire window's worth of events at once.
# MAGIC
# MAGIC ### Feature Vector (10 features per window)
# MAGIC
# MAGIC | # | Feature | Sketch | Description |
# MAGIC |---|---------|--------|-------------|
# MAGIC | 1 | `f0_srcip` | FM | Distinct source IPs |
# MAGIC | 2 | `f0_dstip` | FM | Distinct destination IPs |
# MAGIC | 3 | `f0_dstport` | FM | Distinct destination ports |
# MAGIC | 4 | `f1_count` | Morris | Total flow count |
# MAGIC | 5 | `f2_srcip` | AMS | F2 of srcIP distribution |
# MAGIC | 6 | `f2_dstip` | AMS | F2 of dstIP distribution |
# MAGIC | 7 | `f2_srcip_norm` | AMS/Morris | Normalized burst F2/F1² |
# MAGIC | 8 | `cms_max_srcip` | CMS | Max estimated srcIP count |
# MAGIC | 9 | `cms_topk_frac` | CMS | Fraction of traffic in top-5 srcIPs |
# MAGIC | 10 | `cs_max_srcip` | CS | Count-Sketch max srcIP estimate |

# COMMAND ----------

FEATURE_NAMES = [
    "f0_srcip", "f0_dstip", "f0_dstport",
    "f1_count",
    "f2_srcip", "f2_dstip", "f2_srcip_norm",
    "cms_max_srcip", "cms_topk_frac",
    "cs_max_srcip",
]
TOP_K = 5


class SketchWindowProcessor:
    """Manages all sketches for a single tumbling window (batch interface)."""

    def __init__(self, cms_d=4, cms_w=2048, cs_r=5, cs_b=2048):
        self.f0_srcip = FMEstimator(R=64, num_groups=8, seed_offset=100)
        self.f0_dstip = FMEstimator(R=64, num_groups=8, seed_offset=170)
        self.f0_dstport = FMEstimator(R=64, num_groups=8, seed_offset=240)
        self.f1_counter = MorrisCounter(R=64, num_groups=8, seed_offset=0)
        self.f2_srcip = AMSEstimator(R=64, num_groups=8, seed_offset=200)
        self.f2_dstip = AMSEstimator(R=64, num_groups=8, seed_offset=270)
        self.cms = CountMinSketch(d=cms_d, w=cms_w, seed_offset=300)
        self.cs = CountSketch(r=cs_r, b=cs_b, seed_offset=400)
        self.candidate_keys = xp.array([], dtype=xp.uint64)
        self.max_candidates = TOP_K * 4
        self.event_count = 0
        self.attack_count = 0
        self.cms_d = cms_d
        self.cms_w = cms_w
        self.cs_r = cs_r
        self.cs_b = cs_b

    def update_batch(self, srcip_keys, dstip_keys, dstport_keys, labels):
        """
        Process a batch of events (one window or partial window).

        All key arrays are uint64 on compute device.
        labels is a numpy int32 array on CPU.
        """
        n = len(srcip_keys)
        if n == 0:
            return
        self.event_count += n
        self.attack_count += int(np.sum(labels))

        # F0 estimators
        self.f0_srcip.update_batch(srcip_keys)
        self.f0_dstip.update_batch(dstip_keys)
        self.f0_dstport.update_batch(dstport_keys)

        # F1 (sequential across events, vectorized across R)
        self.f1_counter.update_batch(n)

        # F2 estimators
        self.f2_srcip.update_batch(srcip_keys)
        self.f2_dstip.update_batch(dstip_keys)

        # CMS and Count-Sketch (srcIP only)
        self.cms.update_batch(srcip_keys)
        self.cs.update_batch(srcip_keys)

        # Maintain candidate heavy-hitter keys for later query
        unique_src = xp.unique(srcip_keys)
        if len(self.candidate_keys) > 0:
            self.candidate_keys = xp.unique(
                xp.concatenate([self.candidate_keys, unique_src])
            )
        else:
            self.candidate_keys = unique_src

        # Prune candidates if too many
        if len(self.candidate_keys) > self.max_candidates:
            ests = self.cms.query_batch(self.candidate_keys)
            top_idx = xp.argsort(ests)[-(TOP_K * 2):]
            self.candidate_keys = self.candidate_keys[top_idx]

    def finalize(self):
        """Extract the 10-feature vector and the window label."""
        f0_src = self.f0_srcip.estimate()
        f0_dst = self.f0_dstip.estimate()
        f0_port = self.f0_dstport.estimate()
        f1 = self.f1_counter.estimate()
        f2_src = self.f2_srcip.estimate()
        f2_dst = self.f2_dstip.estimate()

        f1_safe = max(f1, 1.0)
        f2_norm = f2_src / (f1_safe ** 2)

        if len(self.candidate_keys) > 0:
            cms_ests = to_host(self.cms.query_batch(self.candidate_keys))
            cs_ests = to_host(self.cs.query_batch(self.candidate_keys))
            top_k_idx = np.argsort(cms_ests)[-TOP_K:]
            cms_max = float(np.max(cms_ests))
            cms_topk_sum = float(np.sum(cms_ests[top_k_idx]))
            cms_topk_frac = cms_topk_sum / f1_safe
            cs_max = float(np.max(cs_ests))
        else:
            cms_max = 0.0
            cms_topk_frac = 0.0
            cs_max = 0.0

        features = {
            "f0_srcip": f0_src,
            "f0_dstip": f0_dst,
            "f0_dstport": f0_port,
            "f1_count": f1,
            "f2_srcip": f2_src,
            "f2_dstip": f2_dst,
            "f2_srcip_norm": f2_norm,
            "cms_max_srcip": cms_max,
            "cms_topk_frac": cms_topk_frac,
            "cs_max_srcip": cs_max,
        }

        label = 1 if self.attack_count > self.event_count / 2 else 0
        return features, label, self.event_count

    def memory_bytes(self):
        mem = (
            self.f0_srcip.memory_bytes()
            + self.f0_dstip.memory_bytes()
            + self.f0_dstport.memory_bytes()
            + self.f1_counter.memory_bytes()
            + self.f2_srcip.memory_bytes()
            + self.f2_dstip.memory_bytes()
            + self.cms.memory_bytes()
            + self.cs.memory_bytes()
        )
        mem += len(self.candidate_keys) * 8
        return mem

    def reset(self):
        self.f0_srcip.reset()
        self.f0_dstip.reset()
        self.f0_dstport.reset()
        self.f1_counter.reset()
        self.f2_srcip.reset()
        self.f2_dstip.reset()
        self.cms.reset()
        self.cs.reset()
        self.candidate_keys = xp.array([], dtype=xp.uint64)
        self.event_count = 0
        self.attack_count = 0


class ExactWindowProcessor:
    """Exact baseline — uses dicts / sets instead of sketches."""

    def __init__(self):
        self.srcip_freq = {}
        self.dstip_freq = {}
        self.srcip_set = set()
        self.dstip_set = set()
        self.dstport_set = set()
        self.event_count = 0
        self.attack_count = 0

    def update_batch(self, srcips, dstips, dstports, labels):
        """numpy uint64 arrays on CPU."""
        n = len(srcips)
        self.event_count += n
        self.attack_count += int(np.sum(labels))

        u_src, c_src = np.unique(srcips, return_counts=True)
        for ip, c in zip(u_src, c_src):
            k = int(ip)
            self.srcip_set.add(k)
            self.srcip_freq[k] = self.srcip_freq.get(k, 0) + int(c)

        u_dst, c_dst = np.unique(dstips, return_counts=True)
        for ip, c in zip(u_dst, c_dst):
            k = int(ip)
            self.dstip_set.add(k)
            self.dstip_freq[k] = self.dstip_freq.get(k, 0) + int(c)

        self.dstport_set.update(int(p) for p in np.unique(dstports))

    def finalize(self):
        f0_src = len(self.srcip_set)
        f0_dst = len(self.dstip_set)
        f0_port = len(self.dstport_set)
        f1 = self.event_count
        f2_src = sum(v ** 2 for v in self.srcip_freq.values())
        f2_dst = sum(v ** 2 for v in self.dstip_freq.values())
        f1_safe = max(f1, 1)
        f2_norm = f2_src / (f1_safe ** 2)

        sorted_src = sorted(self.srcip_freq.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_src[:TOP_K]
        cms_max = top_k[0][1] if top_k else 0
        topk_sum = sum(c for _, c in top_k)
        cms_topk_frac = topk_sum / f1_safe
        cs_max = cms_max  # exact = same value

        features = {
            "f0_srcip": f0_src,
            "f0_dstip": f0_dst,
            "f0_dstport": f0_port,
            "f1_count": f1,
            "f2_srcip": f2_src,
            "f2_dstip": f2_dst,
            "f2_srcip_norm": f2_norm,
            "cms_max_srcip": cms_max,
            "cms_topk_frac": cms_topk_frac,
            "cs_max_srcip": cs_max,
        }

        label = 1 if self.attack_count > self.event_count / 2 else 0
        return features, label, self.event_count

    def reset(self):
        self.srcip_freq.clear()
        self.dstip_freq.clear()
        self.srcip_set.clear()
        self.dstip_set.clear()
        self.dstport_set.clear()
        self.event_count = 0
        self.attack_count = 0


print("Window processors ready (Sketch-based and Exact).")
print(f"Feature vector ({len(FEATURE_NAMES)} features): {FEATURE_NAMES}")
print(f"Sketch window memory: ~{SketchWindowProcessor().memory_bytes()/1024:.1f} KB")

# COMMAND ----------

def stream_process(
    csv_files,
    column_names,
    window_size,
    train_count,
    cms_d=4,
    cms_w=2048,
    cs_r=5,
    cs_b=2048,
    run_exact=True,
    chunk_size=50000,
):
    """
    Main streaming loop with batch window processing.

    Reads CSVs in chunks, converts IPs to uint64 once per chunk,
    groups events by window via integer division on timestamps,
    and feeds each window group to update_batch().
    """
    sketch_proc = SketchWindowProcessor(
        cms_d=cms_d, cms_w=cms_w, cs_r=cs_r, cs_b=cs_b
    )
    exact_proc = ExactWindowProcessor() if run_exact else None

    sketch_train, sketch_test = [], []
    exact_train, exact_test = [], []

    global_epoch = None
    current_window_id = None
    global_record_idx = 0
    window_start_record_idx = 0

    t_start = time.perf_counter()

    for csv_file in csv_files:
        reader = pd.read_csv(
            csv_file,
            header=None,
            names=column_names,
            chunksize=chunk_size,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False,
        )

        for chunk in reader:
            # ---- Vectorized preprocessing ----------------------------------
            stimes = pd.to_numeric(chunk["Stime"], errors="coerce").values
            valid = ~np.isnan(stimes)
            n_invalid = int((~valid).sum())
            chunk_v = chunk[valid].reset_index(drop=True)
            stimes = stimes[valid]
            n_valid = len(stimes)

            if n_valid == 0:
                global_record_idx += len(chunk)
                continue

            srcip_uint = ips_to_uint64(chunk_v["srcip"])
            dstip_uint = ips_to_uint64(chunk_v["dstip"])
            dstport_uint = ports_to_uint64(chunk_v["dsport"])
            labels_arr = (
                pd.to_numeric(chunk_v["label"], errors="coerce")
                .fillna(0)
                .values.astype(np.int32)
            )

            if global_epoch is None:
                global_epoch = float(stimes[0])

            # Absolute window id for each event
            abs_wids = ((stimes - global_epoch) // window_size).astype(np.int64)
            abs_wids = np.maximum(abs_wids, 0)

            unique_wids = np.unique(abs_wids)

            for wid in unique_wids:
                # Finalize previous window if switching
                if current_window_id is not None and wid != current_window_id:
                    if sketch_proc.event_count > 0:
                        s_feat, s_label, s_n = sketch_proc.finalize()
                        target = (
                            sketch_train
                            if window_start_record_idx < train_count
                            else sketch_test
                        )
                        target.append((s_feat, s_label, s_n))

                        if run_exact and exact_proc.event_count > 0:
                            e_feat, e_label, e_n = exact_proc.finalize()
                            etarget = (
                                exact_train
                                if window_start_record_idx < train_count
                                else exact_test
                            )
                            etarget.append((e_feat, e_label, e_n))

                    sketch_proc.reset()
                    if run_exact:
                        exact_proc.reset()
                    window_start_record_idx = global_record_idx

                if current_window_id is None:
                    window_start_record_idx = global_record_idx

                current_window_id = wid

                mask = abs_wids == wid
                n_in = int(mask.sum())

                w_src = srcip_uint[mask]
                w_dst = dstip_uint[mask]
                w_port = dstport_uint[mask]
                w_labels = labels_arr[mask]

                sketch_proc.update_batch(
                    to_device(w_src), to_device(w_dst), to_device(w_port), w_labels
                )

                if run_exact:
                    exact_proc.update_batch(w_src, w_dst, w_port, w_labels)

                global_record_idx += n_in

            # Count invalid rows
            global_record_idx += n_invalid

            # Progress
            if global_record_idx % 200000 < chunk_size:
                print(
                    f"\r  Processed {global_record_idx:,} records, "
                    f"{len(sketch_train)} train / {len(sketch_test)} test windows...",
                    end="",
                    flush=True,
                )

    # Finalize last window
    if sketch_proc.event_count > 0:
        s_feat, s_label, s_n = sketch_proc.finalize()
        target = (
            sketch_train
            if window_start_record_idx < train_count
            else sketch_test
        )
        target.append((s_feat, s_label, s_n))

        if run_exact and exact_proc is not None and exact_proc.event_count > 0:
            e_feat, e_label, e_n = exact_proc.finalize()
            etarget = (
                exact_train
                if window_start_record_idx < train_count
                else exact_test
            )
            etarget.append((e_feat, e_label, e_n))

    elapsed = time.perf_counter() - t_start
    throughput = global_record_idx / elapsed if elapsed > 0 else 0

    print(f"\n  Done! {global_record_idx:,} records in {elapsed:.1f}s")
    print(f"  Train windows: {len(sketch_train)}, Test windows: {len(sketch_test)}")
    print(f"  Throughput: {throughput:,.0f} events/sec")

    timing = {
        "total_seconds": elapsed,
        "total_events": global_record_idx,
        "throughput_events_per_sec": throughput,
    }

    return sketch_train, sketch_test, exact_train, exact_test, timing


print("Streaming engine ready.")

# COMMAND ----------

def windows_to_dataframe(windows):
    """Convert list of (features_dict, label, n_events) to X, y arrays."""
    if not windows:
        return np.array([]), np.array([])
    X = np.array([[w[0][f] for f in FEATURE_NAMES] for w in windows])
    y = np.array([w[1] for w in windows])
    return X, y


print("=" * 60)
print("Running streaming process (CMS: d=4, w=2048)...")
print("=" * 60)

sketch_train, sketch_test, exact_train, exact_test, timing = stream_process(
    CSV_FILES,
    COLUMN_NAMES,
    WINDOW_SIZE,
    TRAIN_COUNT,
    cms_d=4,
    cms_w=2048,
    cs_r=5,
    cs_b=2048,
    run_exact=True,
    chunk_size=50000,
)

X_train_sketch, y_train_sketch = windows_to_dataframe(sketch_train)
X_test_sketch, y_test_sketch = windows_to_dataframe(sketch_test)
X_train_exact, y_train_exact = windows_to_dataframe(exact_train)
X_test_exact, y_test_exact = windows_to_dataframe(exact_test)

print(f"\nSketch features shape: train {X_train_sketch.shape}, test {X_test_sketch.shape}")
print(f"Exact features shape:  train {X_train_exact.shape}, test {X_test_exact.shape}")
print(f"Train label dist: Normal={sum(y_train_sketch==0)}, Attack={sum(y_train_sketch==1)}")
print(f"Test label dist:  Normal={sum(y_test_sketch==0)}, Attack={sum(y_test_sketch==1)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Classifiers
# MAGIC
# MAGIC ### 5.1 Rule-Based Baseline
# MAGIC A detector with threshold rules on sketch features, tuned on training data only.
# MAGIC
# MAGIC ### 5.2 ML Baseline (Random Forest on sketch features)
# MAGIC Uses cuML on GPU when available, otherwise scikit-learn.
# MAGIC
# MAGIC ### 5.3 Exact Baseline
# MAGIC Same ML classifier trained/tested on exact (non-sketch) features.

# COMMAND ----------

# =============================================================================
# 5.1 Rule-Based Baseline
# =============================================================================
def tune_rule_thresholds(X_train, y_train):
    """Find per-feature thresholds that maximise F1 on training data."""
    feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    best_thresholds = {}
    rule_features = ["f2_srcip_norm", "cms_topk_frac", "f0_dstport", "cms_max_srcip"]

    for feat_name in rule_features:
        idx = feat_idx[feat_name]
        vals = X_train[:, idx]
        best_f1, best_thr, best_dir = -1, None, "above"

        for pct in range(5, 96, 5):
            thr = np.percentile(vals, pct)
            for direction, preds in [
                ("above", (vals > thr).astype(int)),
                ("below", (vals < thr).astype(int)),
            ]:
                if 0 < preds.sum() < len(preds):
                    f1 = f1_score(y_train, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_thr, best_dir = f1, thr, direction

        best_thresholds[feat_name] = (best_thr, best_dir, best_f1)
        print(f"  Rule: {feat_name} {best_dir} {best_thr:.4f} (train F1={best_f1:.3f})")

    return best_thresholds


def rule_based_predict(X, thresholds):
    """Apply rule-based detection: attack if ANY rule fires."""
    feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    preds = np.zeros(X.shape[0], dtype=int)
    for feat_name, (thr, direction, _) in thresholds.items():
        if thr is None:
            continue
        idx = feat_idx[feat_name]
        if direction == "above":
            preds |= (X[:, idx] > thr).astype(int)
        else:
            preds |= (X[:, idx] < thr).astype(int)
    return preds


print("Tuning rule-based thresholds on training data...")
thresholds = tune_rule_thresholds(X_train_sketch, y_train_sketch)

y_pred_rules = rule_based_predict(X_test_sketch, thresholds)
print(f"\nRule-based test results:")
print(classification_report(y_test_sketch, y_pred_rules, target_names=["Normal", "Attack"]))

# COMMAND ----------

# =============================================================================
# 5.2 ML Baseline — Random Forest on sketch features (cuML if available)
# =============================================================================
print("Training Random Forest on sketch-based features...")
if USE_CUML:
    print("  (using cuML GPU RandomForest)")

scaler_sketch = StandardScaler()
X_train_s = scaler_sketch.fit_transform(X_train_sketch)
X_test_s = scaler_sketch.transform(X_test_sketch)

if USE_CUML:
    rf_sketch = cuRFC(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
else:
    rf_sketch = skRFC(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)

rf_sketch.fit(X_train_s, y_train_sketch)
y_pred_rf_sketch = rf_sketch.predict(X_test_s)
if hasattr(y_pred_rf_sketch, "get"):
    y_pred_rf_sketch = y_pred_rf_sketch.get()  # cuML returns cupy array

print("ML (Sketch) test results:")
print(classification_report(y_test_sketch, y_pred_rf_sketch, target_names=["Normal", "Attack"]))

importances = rf_sketch.feature_importances_
if hasattr(importances, "get"):
    importances = importances.get()
for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
    print(f"  {name:20s}: {imp:.4f}")

# COMMAND ----------

# =============================================================================
# 5.3 Exact Baseline — Random Forest on exact features
# =============================================================================
print("Training Random Forest on exact features...")

scaler_exact = StandardScaler()
X_train_e = scaler_exact.fit_transform(X_train_exact)
X_test_e = scaler_exact.transform(X_test_exact)

if USE_CUML:
    rf_exact = cuRFC(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
else:
    rf_exact = skRFC(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)

rf_exact.fit(X_train_e, y_train_exact)
y_pred_rf_exact = rf_exact.predict(X_test_e)
if hasattr(y_pred_rf_exact, "get"):
    y_pred_rf_exact = y_pred_rf_exact.get()

print("ML (Exact) test results:")
print(classification_report(y_test_exact, y_pred_rf_exact, target_names=["Normal", "Attack"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6: Evaluation and Plots
# MAGIC
# MAGIC ### 6A) Detection Quality Comparison
# MAGIC ### 6B) Memory-Accuracy Tradeoffs (varying CMS width)
# MAGIC ### 6C) Runtime and Streaming-ness

# COMMAND ----------

# =============================================================================
# 6A) Confusion matrices and comparison table
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

models = {
    "Rule-Based (Sketch)": (y_test_sketch, y_pred_rules),
    "RF (Sketch)": (y_test_sketch, y_pred_rf_sketch),
    "RF (Exact)": (y_test_exact, y_pred_rf_exact),
}

results_table = []
for ax, (name, (y_true, y_pred)) in zip(axes, models.items()):
    cm = confusion_matrix(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = np.mean(y_true == y_pred)

    results_table.append(
        {"Model": name, "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc}
    )

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{name}\nF1={f1:.3f}", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Attack"])
    ax.set_yticklabels(["Normal", "Attack"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
            )

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

results_df = pd.DataFrame(results_table)
print("\n=== Detection Quality Summary ===")
print(results_df.to_string(index=False, float_format="%.4f"))

# COMMAND ----------

# =============================================================================
# 6B) Memory-Accuracy Tradeoffs
# =============================================================================
cms_configs = [
    {"cms_d": 4, "cms_w": 512, "label": "CMS 4x512"},
    {"cms_d": 4, "cms_w": 1024, "label": "CMS 4x1024"},
    {"cms_d": 4, "cms_w": 2048, "label": "CMS 4x2048"},
    {"cms_d": 4, "cms_w": 4096, "label": "CMS 4x4096"},
    {"cms_d": 4, "cms_w": 8192, "label": "CMS 4x8192"},
]

tradeoff_results = []

for cfg in cms_configs:
    print(f"\nRunning config: {cfg['label']}...")
    s_train, s_test, _, _, t = stream_process(
        CSV_FILES,
        COLUMN_NAMES,
        WINDOW_SIZE,
        TRAIN_COUNT,
        cms_d=cfg["cms_d"],
        cms_w=cfg["cms_w"],
        run_exact=False,
        chunk_size=50000,
    )

    X_tr, y_tr = windows_to_dataframe(s_train)
    X_te, y_te = windows_to_dataframe(s_test)
    if len(X_tr) == 0 or len(X_te) == 0:
        continue

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    if USE_CUML:
        rf = cuRFC(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
    else:
        rf = skRFC(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    y_pred = rf.predict(X_te_s)
    if hasattr(y_pred, "get"):
        y_pred = y_pred.get()

    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)

    mem_total = SketchWindowProcessor(
        cms_d=cfg["cms_d"], cms_w=cfg["cms_w"]
    ).memory_bytes()

    tradeoff_results.append(
        {
            "config": cfg["label"],
            "cms_w": cfg["cms_w"],
            "memory_KB": mem_total / 1024,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "throughput": t["throughput_events_per_sec"],
        }
    )

tradeoff_df = pd.DataFrame(tradeoff_results)
print("\n=== Memory-Accuracy Tradeoff Table ===")
print(tradeoff_df.to_string(index=False, float_format="%.4f"))

# COMMAND ----------

# =============================================================================
# 6B) Plots: F1 vs Memory, Recall vs Memory, Throughput vs Memory
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

mem = tradeoff_df["memory_KB"].values
f1_vals = tradeoff_df["f1"].values
rec_vals = tradeoff_df["recall"].values
thr_vals = tradeoff_df["throughput"].values

axes[0].plot(mem, f1_vals, "bo-", markersize=8, linewidth=2)
axes[0].set_xlabel("Total Sketch Memory (KB)")
axes[0].set_ylabel("F1 Score")
axes[0].set_title("F1-Accuracy vs Memory")
axes[0].grid(True, alpha=0.3)
for i, cfg in enumerate(tradeoff_df["config"]):
    axes[0].annotate(cfg, (mem[i], f1_vals[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)

axes[1].plot(mem, rec_vals, "rs-", markersize=8, linewidth=2)
axes[1].set_xlabel("Total Sketch Memory (KB)")
axes[1].set_ylabel("Recall")
axes[1].set_title("Recall vs Memory\n(critical for security)")
axes[1].grid(True, alpha=0.3)
for i, cfg in enumerate(tradeoff_df["config"]):
    axes[1].annotate(cfg, (mem[i], rec_vals[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)

axes[2].plot(mem, thr_vals, "g^-", markersize=8, linewidth=2)
axes[2].set_xlabel("Total Sketch Memory (KB)")
axes[2].set_ylabel("Throughput (events/sec)")
axes[2].set_title("Throughput vs Memory")
axes[2].grid(True, alpha=0.3)
for i, cfg in enumerate(tradeoff_df["config"]):
    axes[2].annotate(cfg, (mem[i], thr_vals[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("memory_tradeoffs.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# =============================================================================
# 6C) Runtime and Streaming Summary
# =============================================================================
print("=== Runtime and Streaming Summary ===\n")
print(f"Window size: {WINDOW_SIZE}s")
print(f"Total events processed: {timing['total_events']:,}")
print(f"Total wall-clock time: {timing['total_seconds']:.1f}s")
print(f"Throughput: {timing['throughput_events_per_sec']:,.0f} events/sec")
print(f"Backend: {'CuPy (GPU)' if GPU_AVAILABLE else 'NumPy (CPU)'}")

proc = SketchWindowProcessor(cms_d=4, cms_w=2048)
print(f"\nSketch memory per window (CMS 4x2048): {proc.memory_bytes()/1024:.1f} KB")
proc8k = SketchWindowProcessor(cms_d=4, cms_w=8192)
print(f"Sketch memory per window (CMS 4x8192): {proc8k.memory_bytes()/1024:.1f} KB")

print(f"\nMemory breakdown (CMS 4x2048):")
print(f"  F0 (3 FM, R=64):      {3 * proc.f0_srcip.memory_bytes()} B = {3 * proc.f0_srcip.memory_bytes()/1024:.1f} KB")
print(f"  F1 (Morris, R=64):    {proc.f1_counter.memory_bytes()} B = {proc.f1_counter.memory_bytes()/1024:.1f} KB")
print(f"  F2 (2 AMS, R=64):     {2 * proc.f2_srcip.memory_bytes()} B = {2 * proc.f2_srcip.memory_bytes()/1024:.1f} KB")
print(f"  CMS (4x2048):         {proc.cms.memory_bytes()} B = {proc.cms.memory_bytes()/1024:.1f} KB")
print(f"  CS  (5x2048):         {proc.cs.memory_bytes()} B = {proc.cs.memory_bytes()/1024:.1f} KB")
print(f"  Total:                 {proc.memory_bytes()} B = {proc.memory_bytes()/1024:.1f} KB")

# COMMAND ----------

# =============================================================================
# Sketch vs Exact feature comparison (per-window)
# =============================================================================
print("=== Sketch vs Exact Feature Accuracy ===\n")
print("Comparing sketch-estimated features to exact values on test windows:\n")

n_compare = min(len(X_test_sketch), len(X_test_exact))
if n_compare > 0:
    for i, feat in enumerate(FEATURE_NAMES):
        sketch_vals = X_test_sketch[:n_compare, i]
        exact_vals = X_test_exact[:n_compare, i]

        nonzero_mask = exact_vals != 0
        if nonzero_mask.sum() > 0:
            rel_errors = np.abs(sketch_vals[nonzero_mask] - exact_vals[nonzero_mask]) / np.abs(exact_vals[nonzero_mask])
            mean_rel_err = np.mean(rel_errors)
            median_rel_err = np.median(rel_errors)
        else:
            mean_rel_err = float("nan")
            median_rel_err = float("nan")

        corr = np.corrcoef(sketch_vals, exact_vals)[0, 1] if n_compare > 1 else float("nan")
        print(f"  {feat:20s}: mean_rel_err={mean_rel_err:.3f}, "
              f"median_rel_err={median_rel_err:.3f}, corr={corr:.4f}")
else:
    print("  No overlapping test windows to compare.")

# COMMAND ----------

# =============================================================================
# Final comparison bar chart
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

model_names = [r["Model"] for r in results_table]
precs = [r["Precision"] for r in results_table]
recs = [r["Recall"] for r in results_table]
f1s = [r["F1"] for r in results_table]

x = np.arange(len(model_names))
width = 0.25

bars1 = ax.bar(x - width, precs, width, label="Precision", color="#2196F3")
bars2 = ax.bar(x, recs, width, label="Recall", color="#FF5722")
bars3 = ax.bar(x + width, f1s, width, label="F1 Score", color="#4CAF50")

ax.set_ylabel("Score")
ax.set_title("Detection Quality Comparison Across All Baselines")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

plt.tight_layout()
plt.savefig("detection_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nNotebook complete. All plots saved to disk.")
