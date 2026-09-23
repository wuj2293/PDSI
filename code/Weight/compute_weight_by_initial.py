"""
compute_weight_by_initial.py

Builds the corrected WeightByInitial lookup table.

Background
----------
`initial` here is the count of 1s in one binarized 20x20 (= 400-cell)
MaxEnt -> LHS sample -- NOT a fixed field-observed value. Each of the 500
Monte Carlo draws produces its own binarized grid, that grid's 1-count k
(0-399) is looked up in this table (per rule), and only the count matters
for the lookup (not which specific cells are 1), because the table itself
is built by averaging over many random arrangements of k ones.

The original table was built, for every (rule, initial count) pair, from
a SINGLE random arrangement (no repetition), evolved for 400 generations,
with a linear regression fit across the full 0-400 trajectory and its
value at generation 60 taken as the table entry.

Independent validation (see the accompanying Supplementary Note) found
that this global-line-fit step introduces a large, systematic bias --
median 1.9-2.3 and up to ~48 out of the 0-400 scale, across the 77 rules
-- because most of the fitted range lies well past generation 60. Running
with only one arrangement (no repetition) additionally means the original
table carried whatever sampling noise a single random arrangement happens
to have, with nothing to average it out.

This script replaces both: no regression, and N_REPEATS independent
random arrangements averaged per cell.

Usage
-----
    python compute_weight_by_initial.py                  # all 77 rules
    python compute_weight_by_initial.py --rules 8 6 2     # a subset
    python compute_weight_by_initial.py --n-repeats 500   # higher precision
                                                           # for specific
                                                           # noisy cells

Output: WeightByInitial_new_table.csv, columns:
    rule              ECA rule id (0-255)
    initial           count of 1s in the binarized grid, 0-399
    new_mean_gen60    recommended WeightByInitial[rule][initial] value
    new_sd_gen60      SD across the N_REPEATS arrangements
                       (new_sd_gen60 / sqrt(N_REPEATS) = standard error)

Runtime: roughly 4-6 minutes for all 77 rules at N_REPEATS=100 on a single
CPU core (numpy-vectorized per rule). To parallelize, run this script
twice with disjoint --rules subsets in separate processes and concatenate
the two output CSVs.
"""
import argparse
import time
import numpy as np
import pandas as pd

from eca_core import direct_read_table

# The 77 canonical ECA rule identifiers used in the original WeightByInitial table.
RULES_DEFAULT = [8, 6, 2, 34, 38, 12, 10, 14, 26, 30, 58, 62, 74, 78, 106, 110, 40, 44,
                 42, 46, 136, 140, 138, 142, 154, 158, 168, 172, 170, 174, 186, 190,
                 130, 134, 162, 166, 234, 238, 202, 206, 24, 28, 56, 60, 152, 156,
                 184, 188, 4, 18, 22, 36, 50, 54, 72, 76, 90, 94, 104, 108, 122, 126,
                 132, 146, 150, 160, 164, 178, 182, 200, 204, 218, 222, 232, 236,
                 250, 254]
assert len(RULES_DEFAULT) == 77


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", type=int, nargs="+", default=RULES_DEFAULT,
                     help="ECA rule ids to compute (default: all 77 canonical rules)")
    ap.add_argument("--n-repeats", type=int, default=100,
                     help="independent random arrangements averaged per cell (default 100)")
    ap.add_argument("--n-init", type=int, default=400,
                     help="number of initial-count values, 0..n_init-1 (default 400)")
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--out", type=str, default="WeightByInitial_new_table.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    t0 = time.time()
    for i, rule in enumerate(args.rules):
        mean_arr, sd_arr = direct_read_table(rule, args.n_repeats, n_init=args.n_init, rng=rng)
        for k in range(args.n_init):
            rows.append({"rule": rule, "initial": k,
                         "new_mean_gen60": mean_arr[k], "new_sd_gen60": sd_arr[k]})
        print(f"rule {rule} done ({i + 1}/{len(args.rules)}), elapsed {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"saved {len(df)} rows to {args.out} in {time.time() - t0:.1f}s")

    # Optional: reshape into a (n_rules, n_init) matrix, e.g. to save as .npy
    # in whatever array layout the pipeline's clu_SI lookup expects:
    #
    #   mat = df.pivot(index="rule", columns="initial", values="new_mean_gen60")
    #   mat = mat.loc[args.rules].to_numpy()      # shape (len(args.rules), n_init)
    #   np.save("WeightByInitial_new.npy", mat)


if __name__ == "__main__":
    main()