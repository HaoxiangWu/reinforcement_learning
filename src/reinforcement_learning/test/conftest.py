"""
Shared fixtures and helper functions for MDP construction tests.
"""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_grid(rows):
    """Convert a list of strings to a 2D numpy char array.

    Example:
        make_grid(["...", "S#G"])  ->  array([['.', '.', '.'],
                                              ['S', '#', 'G']])
    """
    return np.array([list(row) for row in rows])


def check_probability_matrix(p):
    """Verify every row p[i,a,:] sums to either 1.0 (non-terminal) or 0.0 (terminal).

    Raises AssertionError if any row-sum deviates from 0 or 1.
    """
    row_sums = p.sum(axis=2)
    for i in range(p.shape[0]):
        for a in range(p.shape[1]):
            s = row_sums[i, a]
            assert np.isclose(s, 1.0) or np.isclose(s, 0.0), (
                f"Row ({i},{a}) sums to {s}, expected 0 or 1"
            )


def check_mu(mu):
    """Verify mu sums to 1 and all entries are in [0, 1]."""
    assert np.isclose(mu.sum(), 1.0), f"mu sum = {mu.sum()}, expected 1.0"
    assert np.all(mu >= 0) and np.all(mu <= 1), "mu contains entries outside [0, 1]"


def get_terminal_states(p):
    """Return a set of state indices whose entire transition row is all zeros (absorbing)."""
    return {i for i in range(p.shape[0]) if np.allclose(p[i], 0)}


def get_cell_index(env, coord):
    """Get the state index for a grid (row, col) coordinate.

    Returns None if the coordinate is not in cell_list.
    """
    cell_arr = np.array(env.cell_list)
    matches = np.where((cell_arr == coord).all(axis=1))[0]
    return int(matches[0]) if len(matches) > 0 else None


# ---------------------------------------------------------------------------
# Shared test grids (for use across both test files)
# ---------------------------------------------------------------------------

# --- CliffWalking grids ---

CLIFF_STANDARD = make_grid([
    "....",
    "....",
    "....",
    "S##G",
])

CLIFF_MULTI_START = make_grid([
    "....G",
    "SS###",
])

CLIFF_NO_GOAL = make_grid([
    "....",
    "S##.",
])

CLIFF_MULTI_GOAL = make_grid([
    "G..G",
    "S.#.",
])

CLIFF_1x1 = make_grid(["S"])

CLIFF_1xN = make_grid(["S..G"])

# 'S' is also 'G' — the cell is both start and goal
CLIFF_START_IS_GOAL = make_grid([["S"]])

# Note: we use a 1-element ndarray directly for 1x1 case
# For a 1x1 where we want S=G, we can't do that with strings
CLIFF_S_G_SAME = np.array([["G"]])  # single cell: it's G (terminal); no S — but we test this edge case

# --- GridWorldEnv grids ---

GW_STANDARD = make_grid([
    ".....",
    ".....",
    "###..",
    ".....",
    "G**..",
])

GW_NO_HOLES = make_grid([
    "....",
    "S.#G",
    "....",
])

GW_NO_GOAL = make_grid([
    "....",
    "S.#.",
    ".*..",
])

GW_NO_DOT = make_grid([
    "S*G",
    "###",
])

GW_1x1 = make_grid(["."])

GW_ALL_WALLS = make_grid([
    "###",
    "###",
])


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cliff_standard():
    """A typical CliffWalking grid: 4 rows x 4 cols, row 3 = S##G."""
    from reinforcement_learning import CliffWalking
    return CliffWalking(CLIFF_STANDARD)


@pytest.fixture
def gw_standard():
    """A typical GridWorldEnv: 5x5 grid with walls, holes, goal."""
    from reinforcement_learning import GridWorldEnv
    return GridWorldEnv(GW_STANDARD)
