# WeightByInitial correction (2026-09)

The original `weight_generate.ipynb` / `weight_figure.ipynb` (removed
2026-09-23, see git history if needed) built `WeightByInitial.npy` by, for
each (rule, initial) cell, averaging only **10** random arrangements and
then fitting a **linear regression across the full 0–400 generation
trajectory** to read off a value at generation 60. Independent validation
found this introduces a large, systematic bias (median 1.9–2.3, up to ~48
on the 0–400 scale) because most of the fitted range lies well past
generation 60. Both notebooks are fully superseded by the three scripts
below and have been removed from this folder.

`compute_weight_by_initial.py` replaces this: no regression, and
`N_REPEATS` (default 100) independent random arrangements read directly at
generation 60. Output: `WeightByInitial_new_table.csv`
(`rule, initial, new_mean_gen60, new_sd_gen60`).

`build_regression77_docx.py` regenerates the "Regression77" supplementary
figure (77-rule panel grid) using the same direct-read, no-regression
methodology, for visual/manuscript use.

Both scripts import from `eca_core.py`
(`rule_lut`/`seed_states`/`evolve_step`/`direct_read_table`/`example_trajectories`).
**Update (2026-09-23): received and added.** The file first sent under that
name turned out to be a different, unrelated script (a PDSI
repeated-binarization stability check, see
`code/CNN/CA_77/Jin/pdsi_stability_verification_MISLABELED_as_eca_core.py`
for that content) — the correct `eca_core.py` was requested and re-sent, and
is now in this folder.

**Validation:** ran `compute_weight_by_initial.py` end-to-end (all 77 rules,
n-repeats=100, default seed) and compared all 30,800 (rule, initial) cells
against the `WeightByInitial_new.csv` already in use
(`DATA/weights/WeightByInitial_new.csv`, generated independently by Jin):
mean absolute difference 0.37 (on a 0–400 scale), and only 0.13% of cells
differ by more than 3 combined standard errors — matching the ~0.13-0.27%
expected from Monte-Carlo noise alone. The two are the same table up to
random-seed noise, confirming the three scripts reproduce it correctly.

## Current pipeline status

- The corrected values are already validated and in use as
  `DATA/weights/WeightByInitial_new.csv` (project data folder, outside this
  repo). Its 77 rule-rows were checked to match Jin's
  `WeightByInitial_new_scheme2ready.npy` exactly (max abs diff = 0.0).
- All notebooks under `code/CNN/CA_77/` that compute `type1`
  (cell-wise/independent-per-period binarization) and `type2`
  (global/shared-per-period binarization) now load this corrected table
  instead of the original `WeightByInitial.npy`.
- The original `WeightByInitial.npy` is considered superseded/incorrect and
  should not be used for new results.
