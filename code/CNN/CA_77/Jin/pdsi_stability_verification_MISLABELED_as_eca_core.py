#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDSI Repeated-Binarization Stability Verification
====================================================

WHAT THIS SCRIPT IS
--------------------
This script extends the actual PDSI computation pipeline found in
`PDSI_latin.ipynb` (functions `make_maxent_mat`, `get_binary_maxent_mat`,
`binary_maxent_mat_500`, `make_CA_distribution`, `clu_SI`, reproduced here
verbatim where possible) to run the repeated-binarization stability check
designed in `PDSI_Stability_Check_Protocol.docx` (Sections 4 and 7):

  1. Re-runs the full probability -> binary -> CNN -> SI/PDSI pipeline
     R independent times (new random seed each time), holding the MaxEnt
     probability maps and the trained CNN weights fixed (Section 2 of the
     protocol).
  2. Computes the four stability metrics from Section 4:
       (a) PDSI mean / SD / range across repetitions, per district x pathway
       (b) Red/Orange/Yellow/Green zone-classification stability
       (c) Bottom-10 pathway-set Jaccard overlap across repetitions
       (d) Range of the headline "86.3%" low-emission-share statistic
  3. Implements BOTH binarization designs from Section 7 so they can be
     run side by side:
       - "Scheme 1" (as actually implemented in PDSI_latin.ipynb): the
         reference random matrix is drawn completely independently for
         every (scenario, period, draw) combination.
       - "Scheme 2" (the protocol's proposed alternative): one reference
         random matrix is drawn per draw and REUSED across all 12
         (3 SSP x 4 period) probability layers of a district for that draw.

WHAT THIS SCRIPT IS NOT
------------------------
It is not a replacement for running it. It requires TensorFlow/Keras and
h5py to load the trained `.keras` models, and access to the real
`DATA.zip` contents (sampling.pkl, WeightByInitial.npy, local_index.csv).
Those are not available in the sandboxed environment this script was
written in, so the pipeline-facing code below (`load_models`,
`load_local_lhs`, `load_weight_table`, `make_CA_distribution`) has been
checked against the notebook's source but NOT executed against the real
models. Please treat the first real run as a validation run: sanity-check
a couple of PDSI values against the numbers already in `77_lhs.csv` /
Table 1 before trusting the stability-check output.

Everything that does NOT require TensorFlow (make_maxent_mat, both
binarization schemes, clu_SI, all four stability metrics, the Jaccard/
zone-classification math) IS covered by `--mode selftest` below, which
runs entirely on synthetic data with a numpy-only stand-in "model" and
was executed successfully in the authoring environment. Run it first
after installing dependencies, before pointing the script at real data,
to confirm nothing broke in your Python/numpy version.

ASSUMPTIONS YOU SHOULD CONFIRM BEFORE TRUSTING THE OUTPUT
-----------------------------------------------------------
1. "2040-2060" headline period (Section 4, item 4 of the protocol) is
   mapped to the pathway's 2050 slot (index 1 of the 4-period tuple).
   Change with --headline-period-index if the intended mapping differs.
2. Zone thresholds are taken verbatim from the manuscript text (Red >
   0.825; Orange 0.757-0.825; Yellow 0.721-0.757; Green <= 0.721;
   applied to each district's MINIMUM PDSI across the 81 pathways) and
   are held FIXED across repetitions (i.e. we are checking whether a
   district's classification is stable against a fixed rule, not
   re-deriving new quantile boundaries every repetition). Override with
   --red-cut / --orange-cut / --yellow-cut if the manuscript's numbers
   change in a later revision.
3. The 19 Green-zone district names are not hardcoded anywhere in the
   notebook or DATA.zip, so --districts defaults to 'auto-green', which
   infers them from a baseline (repetition 0, Scheme 1) run using the
   thresholds above. Pass --districts with an explicit comma list (or
   --green-zone-file with one district name per line, matching the
   SIG_ENG_NM spelling in local_index.csv, e.g. Gwangju-si1/Gwangju-si2)
   to use the authors' actual Table 1 list instead -- recommended.
4. "Bottom-10" = the 10 pathways (of 81) with the LOWEST PDSI for a
   district (i.e. the safest 10), matching the manuscript's framing of
   "achieving PDSI values in the bottom 10 of the 81 scenarios" as a
   desirable outcome.

USAGE
-----
    # 1. Structural self-test (no TensorFlow needed, run this first):
    python3 pdsi_stability_verification.py --mode selftest

    # 2. Real run (needs tensorflow, h5py, and DATA.zip extracted somewhere):
    python3 pdsi_stability_verification.py --mode run \\
        --base /path/to/extracted/DATA \\
        --option B --scheme both --R 30 --seed-base 20260921 \\
        --out ./stability_out
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants derived directly from PDSI_latin.ipynb and the manuscript text
# ---------------------------------------------------------------------------

SSPS = ["126", "245", "585"]
PERIODS = ["2030", "2050", "2070", "2090"]

# The 77 representative ECA rule IDs, verbatim from notebook cell 20.
LABELS = [8, 6, 2, 34, 38, 12, 10, 14, 26, 30, 58, 62, 74, 78, 106, 110, 40, 44,
          42, 46, 136, 140, 138, 142, 154, 158, 168, 172, 170, 174, 186, 190,
          130, 134, 162, 166, 234, 238, 202, 206, 24, 28, 56, 60, 152, 156,
          184, 188, 4, 18, 22, 36, 50, 54, 72, 76, 90, 94, 104, 108, 122, 126,
          132, 146, 150, 160, 164, 178, 182, 200, 204, 218, 222, 232, 236,
          250, 254]
assert len(LABELS) == 77

# All 81 = 3^4 scenario pathways, as 4-tuples of SSP codes over
# (2030, 2050, 2070, 2090), matching the notebook's `scenarios` list order
# (itertools.product reproduces the same nested-loop order the notebook
# builds by hand).
ALL_PATHWAYS = list(itertools.product(SSPS, repeat=4))
assert len(ALL_PATHWAYS) == 81

# Zone thresholds, verbatim from the manuscript (paragraph [187]):
#   Red    : min PDSI > 0.825                (5 districts incl. Seoul)
#   Orange : 0.757 < min PDSI <= 0.825        (5 districts)
#   Yellow : 0.721 < min PDSI <= 0.757        (5 districts)
#   Green  : min PDSI <= 0.721                (19 districts)
DEFAULT_RED_CUT = 0.825
DEFAULT_ORANGE_CUT = 0.757
DEFAULT_YELLOW_CUT = 0.721


def pathway_key(pathway):
    """('126','245','585','126') -> '126_245_585_126' (matches notebook/H5 keys)."""
    return "_".join(pathway)


def classify_zone(min_pdsi, red_cut, orange_cut, yellow_cut):
    if min_pdsi > red_cut:
        return "Red"
    if min_pdsi > orange_cut:
        return "Orange"
    if min_pdsi > yellow_cut:
        return "Yellow"
    return "Green"


# ---------------------------------------------------------------------------
# Data loading (real run only -- needs the extracted DATA.zip layout)
# ---------------------------------------------------------------------------

def load_models(base: Path):
    """Load the 5 trained CNN models, exactly as notebook cell 2 does."""
    from tensorflow import keras  # local import: only needed for a real run
    model_dir = [base / "CA_77" / "models" / f"model_{i}.keras" for i in range(1, 6)]
    for p in model_dir:
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
    return [keras.models.load_model(str(p)) for p in model_dir]


def load_local_lhs(base: Path):
    """Load the LHS-sampled MaxEnt data (sampling.pkl) and the district list."""
    import pickle
    with open(base / "SDM_data" / "latin" / "TES_maxent" / "sampling.pkl", "rb") as f:
        local_lhs = pickle.load(f)
    local_index = pd.read_csv(base / "SDM_data" / "latin" / "local_index.csv")
    local_name = local_index["SIG_ENG_NM"].tolist()
    return local_lhs, local_name


def load_weight_table(base: Path):
    return np.load(base / "weights" / "WeightByInitial.npy", allow_pickle=True)


# ---------------------------------------------------------------------------
# Core pipeline: MaxEnt matrix assembly (verbatim logic from notebook cell 7)
# ---------------------------------------------------------------------------

def make_maxent_mat(local_lhs, pathway, loc):
    """pathway: 4-tuple of SSP codes over (2030,2050,2070,2090).
    Returns shape (1,20,20,4), channel order = period order."""
    maxent_mat = []
    for ssp, period in zip(pathway, PERIODS):
        key = f"ssp{ssp}_{period}"
        dataF = local_lhs[key][loc]
        dataF = dataF.iloc[:, 2].values
        dataF = np.reshape(dataF, (20, 20))
        maxent_mat.append(dataF)
    maxent_mat = np.array(maxent_mat)                    # (4,20,20)
    maxent_mat = np.transpose(maxent_mat, (1, 2, 0))       # (20,20,4)
    return np.reshape(maxent_mat, (1, 20, 20, 4)).astype(np.float32)


def make_all_period_layers(local_lhs, loc):
    """All 12 (ssp,period) probability layers for a district, needed by
    Scheme 2 (which shares one reference draw across all 12 layers).
    Returns dict[(ssp,period)] -> (20,20) float array."""
    layers = {}
    for ssp in SSPS:
        for period in PERIODS:
            key = f"ssp{ssp}_{period}"
            dataF = local_lhs[key][loc].iloc[:, 2].values
            layers[(ssp, period)] = np.reshape(dataF, (20, 20)).astype(np.float32)
    return layers


# ---------------------------------------------------------------------------
# Binarization -- Scheme 1 (as actually implemented) and Scheme 2 (proposed)
# ---------------------------------------------------------------------------

def binarize_scheme1_batch(maxent_mat, n, rng):
    """Scheme 1: fully independent reference draw per sample, per pixel,
    per channel. maxent_mat: (1,20,20,4). Returns (n,20,20,4) binary array.
    This is exactly `np.random.rand(n,20,20,4) < maxent_mat`, the line
    used inside notebook cell 18 (make_CA_distribution), generalized to an
    injectable Generator so repetitions can use independent seeds."""
    r = rng.random((n, 20, 20, 4))
    return (r < maxent_mat).astype(np.float32)


def binarize_scheme2_layers(period_layers, n, rng):
    """Scheme 2: one shared (20,20) reference draw per sample, reused across
    all 12 (ssp,period) layers of the district for that sample (protocol
    Section 7.1/7.2: draw Rij indexed only by (i,j,k), broadcast over s,t).

    Returns dict[(ssp,period)] -> (n,20,20) binary array, so pathway-level
    (n,20,20,4) tensors can be assembled cheaply afterwards by picking 4 of
    the 12 keys per pathway -- this is what makes Scheme 2 the cheaper
    binarization scheme the protocol notes (Section 7.2, item 2).
    """
    ref = rng.random((n, 20, 20))  # ONE shared draw per sample, reused below
    out = {}
    for key, M in period_layers.items():
        out[key] = (ref < M[None, :, :]).astype(np.float32)
    return out


def assemble_pathway_from_layers(binary_layers, pathway):
    """pathway: 4-tuple of SSP codes. binary_layers: dict[(ssp,period)] ->
    (n,20,20). Returns (n,20,20,4)."""
    stacks = [binary_layers[(ssp, period)] for ssp, period in zip(pathway, PERIODS)]
    return np.stack(stacks, axis=-1)


# ---------------------------------------------------------------------------
# CNN inference + SI (verbatim logic from notebook cells 18 and 24)
# ---------------------------------------------------------------------------

def run_cnn_ensemble(binary_samples, models, weight):
    """binary_samples: (500,20,20,4) already-binarized draws (any scheme).
    Mirrors notebook cell 18: split into 5 batches of 100 (one per loaded
    model), sum softmax outputs and channel-0 cell counts, then average.
    Returns (initial:int, CA_distribution: (77,) float array)."""
    n_total = binary_samples.shape[0]
    n_models = len(models)
    assert n_total % n_models == 0, "expects n_total divisible by number of models"
    batch = n_total // n_models

    b = np.zeros((1, 77), dtype=np.float32)
    initial_sum = 0.0
    for m_idx, model_call in enumerate(models):
        chunk = binary_samples[m_idx * batch:(m_idx + 1) * batch]
        initial_sum += np.sum(chunk[:, :, :, 0])
        model_output = np.array(model_call(chunk), dtype=np.float32)  # (batch,77)
        b += np.sum(model_output, axis=0, keepdims=True)

    initial = int(round(initial_sum / n_total))
    initial = min(max(initial, 0), weight.shape[1] - 1)  # clip into table range
    CA_distribution = (b / n_total).reshape(-1)  # (77,)
    return initial, CA_distribution


def clu_SI(initial, CA_distribution, weight, labels=LABELS):
    SI = 0.0
    for k, rule_id in enumerate(labels):
        SI += CA_distribution[k] * weight[rule_id][initial]
    return SI / 400.0


# ---------------------------------------------------------------------------
# One full PDSI computation for one district x one pathway x one scheme
# ---------------------------------------------------------------------------

def compute_pdsi(local_lhs, models, weight, loc, pathway, scheme, rng,
                  n_draws=500, precomputed_scheme2_layers=None):
    if scheme == "scheme1":
        maxent_mat = make_maxent_mat(local_lhs, pathway, loc)
        binary_samples = binarize_scheme1_batch(maxent_mat, n_draws, rng)
    elif scheme == "scheme2":
        if precomputed_scheme2_layers is None:
            period_layers = make_all_period_layers(local_lhs, loc)
            precomputed_scheme2_layers = binarize_scheme2_layers(period_layers, n_draws, rng)
        binary_samples = assemble_pathway_from_layers(precomputed_scheme2_layers, pathway)
    else:
        raise ValueError(scheme)

    initial, CA_distribution = run_cnn_ensemble(binary_samples, models, weight)
    return clu_SI(initial, CA_distribution, weight)


def run_district_all_pathways(local_lhs, models, weight, loc, scheme, rng, n_draws=500):
    """Returns dict[pathway_key] -> PDSI, for all 81 pathways, one district,
    one binarization scheme, one repetition (one rng state).
    Scheme 2 shares the 12 base binarized layers across all 81 pathways for
    efficiency, exactly as the protocol recommends."""
    result = {}
    if scheme == "scheme2":
        period_layers = make_all_period_layers(local_lhs, loc)
        shared_layers = binarize_scheme2_layers(period_layers, n_draws, rng)
        for pathway in ALL_PATHWAYS:
            pdsi = compute_pdsi(local_lhs, models, weight, loc, pathway, "scheme2",
                                 rng, n_draws, precomputed_scheme2_layers=shared_layers)
            result[pathway_key(pathway)] = pdsi
    else:
        for pathway in ALL_PATHWAYS:
            pdsi = compute_pdsi(local_lhs, models, weight, loc, pathway, "scheme1", rng, n_draws)
            result[pathway_key(pathway)] = pdsi
    return result


# ---------------------------------------------------------------------------
# Stability metrics (protocol Section 4)
# ---------------------------------------------------------------------------

def bottom10_pathways(pdsi_by_pathway, n=10):
    """Lowest-n PDSI pathways for one district in one repetition."""
    ordered = sorted(pdsi_by_pathway.items(), key=lambda kv: kv[1])
    return [k for k, _ in ordered[:n]]


def headline_low_emission_share(bottom10_by_district, headline_period_index=1):
    """Section 4 item 4: among the pooled bottom-10 entries across the
    selected (Green-zone) districts, what fraction have SSP1-2.6 or
    SSP2-4.5 at the headline period (default: the 2050 slot, standing in
    for "2040-2060")?"""
    total = 0
    hits = 0
    for loc, bottoms in bottom10_by_district.items():
        for pkey in bottoms:
            ssp_at_period = pkey.split("_")[headline_period_index]
            total += 1
            if ssp_at_period in ("126", "245"):
                hits += 1
    return hits / total if total else float("nan"), hits, total


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pairwise_mean_jaccard(list_of_sets):
    pairs = list(itertools.combinations(range(len(list_of_sets)), 2))
    if not pairs:
        return float("nan")
    vals = [jaccard(list_of_sets[i], list_of_sets[j]) for i, j in pairs]
    return float(np.mean(vals))


def summarize_repetitions(all_reps_pdsi, districts, red_cut, orange_cut, yellow_cut,
                           headline_period_index):
    """all_reps_pdsi: list (len R) of dict[loc] -> dict[pathway_key] -> PDSI.
    Returns a dict of DataFrames/summary values implementing the four
    Section-4 metrics."""
    R = len(all_reps_pdsi)

    # (a) PDSI mean / SD / range per district x pathway
    rows = []
    for loc in districts:
        for pathway in ALL_PATHWAYS:
            pkey = pathway_key(pathway)
            vals = [all_reps_pdsi[r][loc][pkey] for r in range(R)]
            rows.append({
                "district": loc, "pathway": pkey,
                "pdsi_mean": float(np.mean(vals)), "pdsi_sd": float(np.std(vals, ddof=1) if R > 1 else 0.0),
                "pdsi_min": float(np.min(vals)), "pdsi_max": float(np.max(vals)),
            })
    pdsi_variation = pd.DataFrame(rows)

    # (b) Zone-classification stability
    zone_rows = []
    for loc in districts:
        zones = []
        for r in range(R):
            min_pdsi = min(all_reps_pdsi[r][loc].values())
            zones.append(classify_zone(min_pdsi, red_cut, orange_cut, yellow_cut))
        modal_zone = pd.Series(zones).mode().iloc[0]
        stability = zones.count(modal_zone) / R
        zone_rows.append({"district": loc, "modal_zone": modal_zone,
                           "zone_stability_frac": stability, "zones_seen": sorted(set(zones))})
    zone_stability = pd.DataFrame(zone_rows)

    # (c) Bottom-10 Jaccard overlap per district, across repetitions
    jaccard_rows = []
    for loc in districts:
        bottom_sets = [set(bottom10_pathways(all_reps_pdsi[r][loc])) for r in range(R)]
        jaccard_rows.append({"district": loc, "bottom10_mean_pairwise_jaccard":
                              pairwise_mean_jaccard(bottom_sets)})
    bottom10_jaccard = pd.DataFrame(jaccard_rows)

    # (d) Headline statistic range across repetitions (pooled over districts)
    headline_vals = []
    for r in range(R):
        bottom10_by_district = {loc: bottom10_pathways(all_reps_pdsi[r][loc]) for loc in districts}
        share, hits, total = headline_low_emission_share(bottom10_by_district, headline_period_index)
        headline_vals.append(share)
    headline_summary = {
        "R": R, "mean": float(np.mean(headline_vals)), "sd": float(np.std(headline_vals, ddof=1) if R > 1 else 0.0),
        "min": float(np.min(headline_vals)), "max": float(np.max(headline_vals)),
        "per_repetition": headline_vals,
    }

    return {
        "pdsi_variation": pdsi_variation,
        "zone_stability": zone_stability,
        "bottom10_jaccard": bottom10_jaccard,
        "headline_summary": headline_summary,
    }


# ---------------------------------------------------------------------------
# District-set selection (protocol Section 3: Options A / B / C)
# ---------------------------------------------------------------------------

def select_districts(option, all_district_names, local_lhs, models, weight,
                      red_cut, orange_cut, yellow_cut, explicit=None, seed_base=0):
    if explicit is not None:
        return explicit
    if option == "A":
        return ["Yeoju-si"] if "Yeoju-si" in all_district_names else all_district_names[:1]
    if option == "C":
        return list(all_district_names)
    # Option B (default): "auto-green" -- infer the Green zone from one
    # baseline Scheme-1 run. This is an approximation; pass --districts or
    # --green-zone-file with the authors' actual Table 1 list for the
    # authoritative version (see module docstring, assumption 3).
    print("[select_districts] Inferring Green-zone districts from a baseline "
          "Scheme-1 run (pass --green-zone-file to use the authors' real list instead)...",
          file=sys.stderr)
    rng = np.random.default_rng(seed_base)
    green = []
    for loc in all_district_names:
        pdsi_by_pathway = run_district_all_pathways(local_lhs, models, weight, loc, "scheme1", rng)
        min_pdsi = min(pdsi_by_pathway.values())
        if classify_zone(min_pdsi, red_cut, orange_cut, yellow_cut) == "Green":
            green.append(loc)
    return green


# ---------------------------------------------------------------------------
# Self-test (no TensorFlow required) -- validates all the numpy-side logic
# ---------------------------------------------------------------------------

def _make_fake_model(rule_bias_seed):
    """A deterministic numpy stand-in for a trained keras model: takes a
    (batch,20,20,4) binary array, returns a (batch,77) softmax-like output.
    Not meant to be realistic -- only to exercise run_cnn_ensemble/clu_SI's
    shapes and arithmetic without needing an actual trained network."""
    rng = np.random.default_rng(rule_bias_seed)
    rule_bias = rng.normal(size=77)

    def fake_model(batch):
        # simple deterministic function of the input + fixed per-rule bias
        feat = batch.mean(axis=(1, 2, 3))  # (batch,)
        logits = feat[:, None] * 3.0 + rule_bias[None, :]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return fake_model


def _make_synthetic_local_lhs(district_names, seed=0):
    rng = np.random.default_rng(seed)
    local_lhs = {}
    for ssp in SSPS:
        for period in PERIODS:
            key = f"ssp{ssp}_{period}"
            local_lhs[key] = {}
            for loc in district_names:
                # base probability level depends on (ssp,period,loc) only,
                # so re-querying the same layer is deterministic (needed to
                # correctly test that scheme1 and scheme2 use the SAME
                # underlying probabilities, differing only in RNG sharing).
                h = abs(hash((ssp, period, loc))) % 1000
                base_p = 0.15 + 0.5 * (h / 1000.0)
                probs = np.clip(rng.normal(loc=base_p, scale=0.05, size=400), 0.01, 0.99)
                df = pd.DataFrame({"x": np.arange(400), "y": np.arange(400), "p": probs,
                                    "SIG_ENG_NM": loc})
                local_lhs[key][loc] = df
    return local_lhs


def selftest():
    print("=== SELF-TEST (synthetic data, no TensorFlow required) ===")
    rng_master = np.random.default_rng(42)
    district_names = ["Yeoju-si", "Gwangju-si1", "Gwangju-si2", "Icheon-si"]
    local_lhs = _make_synthetic_local_lhs(district_names, seed=1)
    models = [_make_fake_model(seed) for seed in range(5)]
    weight = rng_master.uniform(0, 400, size=(256, 400)).astype(np.float32)

    # --- 1. make_maxent_mat shape / range checks -------------------------
    pathway = ALL_PATHWAYS[0]
    mm = make_maxent_mat(local_lhs, pathway, "Yeoju-si")
    assert mm.shape == (1, 20, 20, 4), mm.shape
    assert (mm >= 0).all() and (mm <= 1).all()
    print("[OK] make_maxent_mat shape/range")

    # --- 2. Scheme 1 vs Scheme 2 transition-probability check ------------
    # This directly tests the theoretical claim in protocol Section 7.1:
    # for two layers with the SAME true probability p everywhere, Scheme 1
    # should give P(flip) ~= 2p(1-p), Scheme 2 should give P(flip) ~= 0
    # (since same p means literally the same binary outcome under a shared
    # reference draw). For layers with different probabilities p1,p2,
    # Scheme 2's flip rate should match |p1-p2| closely; Scheme 1's should
    # not.
    p_same = 0.4
    M_a = np.full((20, 20), p_same, dtype=np.float32)
    M_b = np.full((20, 20), p_same, dtype=np.float32)
    n_mc = 20000
    rng = np.random.default_rng(7)
    draws_a1 = binarize_scheme1_batch(M_a[None, :, :, None].repeat(4, axis=-1), n_mc, rng)[:, :, :, 0]
    rng = np.random.default_rng(7)  # fresh rng, independent of draws_a1's own consumption pattern
    rng2 = np.random.default_rng(8)
    draws_b1 = binarize_scheme1_batch(M_b[None, :, :, None].repeat(4, axis=-1), n_mc, rng2)[:, :, :, 0]
    flip_rate_scheme1_samep = float(np.mean(draws_a1 != draws_b1))
    theoretical_scheme1_samep = 2 * p_same * (1 - p_same)
    print(f"[check] Scheme1, equal p={p_same}: empirical flip rate={flip_rate_scheme1_samep:.4f}, "
          f"theory 2p(1-p)={theoretical_scheme1_samep:.4f}")
    assert abs(flip_rate_scheme1_samep - theoretical_scheme1_samep) < 0.02

    layers_same = {("126", "2030"): M_a, ("245", "2030"): M_b}
    for k in [("126", p) for p in PERIODS[1:]] + [("585", p) for p in PERIODS]:
        layers_same[k] = M_a  # fill remaining 10 layers arbitrarily, unused below
    rng = np.random.default_rng(9)
    shared = binarize_scheme2_layers(layers_same, n_mc, rng)
    flip_rate_scheme2_samep = float(np.mean(shared[("126", "2030")] != shared[("245", "2030")]))
    print(f"[check] Scheme2, equal p={p_same}: empirical flip rate={flip_rate_scheme2_samep:.4f} "
          f"(theory: 0.0, since a shared draw against equal probabilities always agrees)")
    assert flip_rate_scheme2_samep < 0.005

    p1, p2 = 0.3, 0.7
    layers_diff = dict(layers_same)
    layers_diff[("126", "2030")] = np.full((20, 20), p1, dtype=np.float32)
    layers_diff[("245", "2030")] = np.full((20, 20), p2, dtype=np.float32)
    rng = np.random.default_rng(10)
    shared_diff = binarize_scheme2_layers(layers_diff, n_mc, rng)
    flip_rate_scheme2_diffp = float(np.mean(shared_diff[("126", "2030")] != shared_diff[("245", "2030")]))
    print(f"[check] Scheme2, p1={p1},p2={p2}: empirical flip rate={flip_rate_scheme2_diffp:.4f}, "
          f"theory |p1-p2|={abs(p1-p2):.4f}")
    assert abs(flip_rate_scheme2_diffp - abs(p1 - p2)) < 0.02
    print("[OK] Scheme 1 / Scheme 2 transition-probability formulas confirmed on synthetic data")

    # --- 3. Full pipeline shape check (CNN ensemble + SI) -----------------
    rng = np.random.default_rng(123)
    pdsi_by_pathway_s1 = run_district_all_pathways(local_lhs, models, weight, "Yeoju-si",
                                                     "scheme1", rng, n_draws=50)
    assert len(pdsi_by_pathway_s1) == 81
    assert all(np.isfinite(v) for v in pdsi_by_pathway_s1.values())
    rng = np.random.default_rng(123)
    pdsi_by_pathway_s2 = run_district_all_pathways(local_lhs, models, weight, "Yeoju-si",
                                                     "scheme2", rng, n_draws=50)
    assert len(pdsi_by_pathway_s2) == 81
    print("[OK] Full pipeline runs end-to-end for both schemes, 81/81 pathways, finite PDSI values")

    # --- 4. Stability-metric machinery, using a few cheap repetitions -----
    R = 4
    all_reps = []
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        rep = {}
        for loc in district_names:
            rep[loc] = run_district_all_pathways(local_lhs, models, weight, loc, "scheme1",
                                                   rng, n_draws=20)
        all_reps.append(rep)
    summary = summarize_repetitions(all_reps, district_names, DEFAULT_RED_CUT,
                                     DEFAULT_ORANGE_CUT, DEFAULT_YELLOW_CUT,
                                     headline_period_index=1)
    assert len(summary["pdsi_variation"]) == len(district_names) * 81
    assert len(summary["zone_stability"]) == len(district_names)
    assert len(summary["bottom10_jaccard"]) == len(district_names)
    assert 0.0 <= summary["headline_summary"]["mean"] <= 1.0
    print("[OK] summarize_repetitions produces all four Section-4 metrics with correct shapes")
    print(json.dumps({"headline_summary": summary["headline_summary"]}, indent=2))

    print("=== SELF-TEST PASSED ===")


# ---------------------------------------------------------------------------
# Real run driver
# ---------------------------------------------------------------------------

def real_run(args):
    base = Path(args.base)
    print(f"Loading models from {base} ...", file=sys.stderr)
    models = load_models(base)
    local_lhs, all_district_names = load_local_lhs(base)
    weight = load_weight_table(base)

    explicit = None
    if args.green_zone_file:
        explicit = [l.strip() for l in Path(args.green_zone_file).read_text().splitlines() if l.strip()]
    elif args.districts:
        explicit = [d.strip() for d in args.districts.split(",") if d.strip()]

    districts = select_districts(args.option, all_district_names, local_lhs, models, weight,
                                  args.red_cut, args.orange_cut, args.yellow_cut,
                                  explicit=explicit, seed_base=args.seed_base)
    print(f"Districts selected ({len(districts)}): {districts}", file=sys.stderr)

    schemes = ["scheme1", "scheme2"] if args.scheme == "both" else [args.scheme]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scheme in schemes:
        print(f"--- Running {args.R} repetitions, scheme={scheme} ---", file=sys.stderr)
        all_reps = []
        for r in range(args.R):
            seed = args.seed_base + r
            rng = np.random.default_rng(seed)
            rep = {}
            for loc in districts:
                rep[loc] = run_district_all_pathways(local_lhs, models, weight, loc, scheme,
                                                       rng, n_draws=500)
            all_reps.append(rep)
            print(f"  repetition {r+1}/{args.R} (seed={seed}) done", file=sys.stderr)

        # raw PDSI values, for full auditability
        raw_rows = []
        for r, rep in enumerate(all_reps):
            for loc, pdsi_by_pathway in rep.items():
                for pkey, pdsi in pdsi_by_pathway.items():
                    raw_rows.append({"repetition": r, "seed": args.seed_base + r,
                                      "district": loc, "pathway": pkey, "pdsi": pdsi})
        pd.DataFrame(raw_rows).to_csv(out_dir / f"raw_pdsi_{scheme}.csv", index=False)

        summary = summarize_repetitions(all_reps, districts, args.red_cut, args.orange_cut,
                                         args.yellow_cut, args.headline_period_index)
        summary["pdsi_variation"].to_csv(out_dir / f"pdsi_variation_{scheme}.csv", index=False)
        summary["zone_stability"].to_csv(out_dir / f"zone_stability_{scheme}.csv", index=False)
        summary["bottom10_jaccard"].to_csv(out_dir / f"bottom10_jaccard_{scheme}.csv", index=False)
        with open(out_dir / f"headline_summary_{scheme}.json", "w") as f:
            json.dump(summary["headline_summary"], f, indent=2)
        print(f"[{scheme}] headline stat: mean={summary['headline_summary']['mean']:.3f}, "
              f"range=[{summary['headline_summary']['min']:.3f}, {summary['headline_summary']['max']:.3f}]",
              file=sys.stderr)

    if args.scheme == "both":
        # direct Scheme1-vs-Scheme2 comparison (protocol Section 7.2 item 4)
        s1 = pd.read_csv(out_dir / "pdsi_variation_scheme1.csv").set_index(["district", "pathway"])["pdsi_mean"]
        s2 = pd.read_csv(out_dir / "pdsi_variation_scheme2.csv").set_index(["district", "pathway"])["pdsi_mean"]
        diff = (s1 - s2).rename("scheme1_minus_scheme2_pdsi_mean")
        diff.to_csv(out_dir / "scheme1_vs_scheme2_pdsi_diff.csv")
        print(f"Scheme1-vs-Scheme2 PDSI mean diff: mean={diff.mean():.5f}, "
              f"SD={diff.std():.5f}, max abs={diff.abs().max():.5f}", file=sys.stderr)

    print(f"All outputs written to {out_dir}", file=sys.stderr)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["selftest", "run"], default="selftest")
    p.add_argument("--base", type=str, help="Path to extracted DATA.zip root "
                   "(containing CA_77/, SDM_data/, weights/, Results/)")
    p.add_argument("--option", choices=["A", "B", "C"], default="B")
    p.add_argument("--districts", type=str, default=None,
                   help="Explicit comma-separated district list, overrides --option")
    p.add_argument("--green-zone-file", type=str, default=None,
                   help="Path to a file with one district name per line (Table 1's actual "
                        "Green-zone list); overrides --option and --districts")
    p.add_argument("--scheme", choices=["scheme1", "scheme2", "both"], default="both")
    p.add_argument("--R", type=int, default=30, help="Number of independent repetitions")
    p.add_argument("--seed-base", type=int, default=20260921)
    p.add_argument("--headline-period-index", type=int, default=1,
                   help="Index into (2030,2050,2070,2090) standing in for the '2040-2060' "
                        "headline period; default 1 = the 2050 slot")
    p.add_argument("--red-cut", type=float, default=DEFAULT_RED_CUT)
    p.add_argument("--orange-cut", type=float, default=DEFAULT_ORANGE_CUT)
    p.add_argument("--yellow-cut", type=float, default=DEFAULT_YELLOW_CUT)
    p.add_argument("--out", type=str, default="./stability_out")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.mode == "selftest":
        selftest()
    else:
        if not args.base:
            print("ERROR: --base is required for --mode run", file=sys.stderr)
            sys.exit(1)
        real_run(args)