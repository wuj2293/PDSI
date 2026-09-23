"""
eca_core.py

Standard-Wolfram-convention elementary cellular automaton (ECA) simulation
core, used to (re)build the WeightByInitial lookup table and the
corresponding Supplementary Data 1 ("Regression77") figure.

Convention
----------
- A 1D array of L cells (L = 400, matching the 20x20 = 400-cell MaxEnt/LHS
  grid). Cells beyond the array boundary are treated as 0 ("zero"
  boundary condition).
- For neighbourhood (left, center, right), the lookup index is
      index = 4*left + 2*center + right      (left = most significant bit)
  and the new state of the center cell is bit `index` of the rule's 8-bit
  number (0-255).

This module has no side effects: it only defines the simulation building
blocks used by compute_weight_by_initial.py and build_regression77_docx.py.
"""
import numpy as np

L = 400          # array length
N_GEN = 400      # generations simulated (matches the original 0-400 trajectory range)
READ_GEN = 60    # generation at which the lookup value is read


def rule_lut(rule: int) -> np.ndarray:
    """8-element lookup table: lut[pattern] = next state of the center cell."""
    return np.array([(rule >> p) & 1 for p in range(8)], dtype=np.uint8)


def seed_states(init_counts: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:
    """
    Vectorized random seeding.

    init_counts : 1D int array of length n_rows. init_counts[r] ones are
                  placed at uniformly random positions in row r (all other
                  cells 0).
    Returns a uint8 array of shape (n_rows, L).
    """
    n_rows = len(init_counts)
    rand = rng.random((n_rows, L))
    # per-row random rank of each cell (0..L-1); the k lowest-rank cells become 1s
    ranks = np.argsort(np.argsort(rand, axis=1), axis=1)
    return (ranks < init_counts[:, None]).astype(np.uint8)


def evolve_step(state: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Advance every row of `state`, shape (n_rows, L), by one ECA generation."""
    n_rows, Lc = state.shape
    padded = np.zeros((n_rows, Lc + 2), dtype=np.uint8)
    padded[:, 1:-1] = state
    left, center, right = padded[:, :-2], padded[:, 1:-1], padded[:, 2:]
    pattern = left.astype(np.int64) * 4 + center * 2 + right
    return lut[pattern]


def direct_read_table(rule: int, n_repeats: int, n_init: int = 400,
                       read_gen: int = READ_GEN, rng: np.random.Generator = None):
    """
    For ONE rule and every initial count 0..n_init-1: seed n_repeats
    independent random arrangements of that many 1s, evolve to `read_gen`
    generations, and return the per-initial-value mean and standard
    deviation of the resulting cell count.

    This is the corrected method: a direct read at generation `read_gen`,
    with NO linear regression / line fitting involved.

    Returns
    -------
    mean : ndarray, shape (n_init,)  -- recommended WeightByInitial value
    sd   : ndarray, shape (n_init,)  -- SD across the n_repeats arrangements;
                                        sd / sqrt(n_repeats) is the standard
                                        error of `mean`.
    """
    if rng is None:
        rng = np.random.default_rng()
    init_per_row = np.repeat(np.arange(n_init), n_repeats)
    lut = rule_lut(rule)
    state = seed_states(init_per_row, L, rng)
    for _ in range(read_gen):
        state = evolve_step(state, lut)
    counts = state.sum(axis=1).reshape(n_init, n_repeats)
    return counts.mean(axis=1), counts.std(axis=1, ddof=1)


def example_trajectories(rule: int, init_values, n_repeats: int, n_gen: int = N_GEN,
                          rng: np.random.Generator = None) -> np.ndarray:
    """
    For ONE rule and a short list of initial values (e.g. [100, 200, 300]),
    return the full 0..n_gen generation mean trajectory for each, as an
    array of shape (len(init_values), n_gen + 1).

    Used only to draw the Supplementary Data 1 example panels (illustrative
    initial values); the production lookup table itself only needs
    direct_read_table(), which is far cheaper because it stops at
    generation `read_gen` instead of simulating the full 0-400 range.
    """
    if rng is None:
        rng = np.random.default_rng()
    init_values = np.asarray(init_values)
    init_per_row = np.repeat(init_values, n_repeats)
    lut = rule_lut(rule)
    state = seed_states(init_per_row, L, rng)
    traj = np.zeros((len(init_values) * n_repeats, n_gen + 1), dtype=np.int32)
    traj[:, 0] = state.sum(axis=1)
    for g in range(1, n_gen + 1):
        state = evolve_step(state, lut)
        traj[:, g] = state.sum(axis=1)
    return traj.reshape(len(init_values), n_repeats, n_gen + 1).mean(axis=1)