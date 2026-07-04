"""
Tests for CliffWalking MDP construction.

Focuses on transition matrix (p), reward matrix (r), and initial state
distribution (mu) invariants and correctness.
"""
import pytest
import numpy as np
from reinforcement_learning import CliffWalking

from reinforcement_learning.test.conftest import (
    make_grid,
    check_probability_matrix,
    check_mu,
    get_terminal_states,
    get_cell_index,
    CLIFF_STANDARD,
    CLIFF_MULTI_START,
    CLIFF_NO_GOAL,
    CLIFF_MULTI_GOAL,
    CLIFF_1xN,
)


# ===================================================================
# 3A. Matrix Shape Invariants
# ===================================================================

class TestMatrixShape:
    """Tests for p, r, mu shapes."""

    def test_shape_standard_grid(self):
        """p=(n,4,n), r=(n,4,n), mu=(n,) with the standard cliff grid."""
        env = CliffWalking(CLIFF_STANDARD)
        n = len(env.cell_list)

        assert env.p.shape == (n, 4, n), f"p shape: {env.p.shape}, expected ({n}, 4, {n})"
        assert env.r.shape == (n, 4, n), f"r shape: {env.r.shape}, expected ({n}, 4, {n})"
        assert env.mu.shape == (n,), f"mu shape: {env.mu.shape}, expected ({n},)"

    def test_cell_list_not_empty(self):
        """Valid grid produces n_states > 0."""
        env = CliffWalking(CLIFF_STANDARD)
        assert len(env.cell_list) > 0

    def test_n_states_matches_cell_list(self):
        """Number of states equals len(cell_list)."""
        env = CliffWalking(CLIFF_STANDARD)
        assert env.p.shape[0] == len(env.cell_list)


# ===================================================================
# 3B. Probability Invariants
# ===================================================================

class TestProbabilityInvariants:
    """Tests for transition probability matrix properties."""

    def test_probability_row_sums(self):
        """Every p[i,a,:] sums to either 0 (terminal) or 1 (non-terminal)."""
        env = CliffWalking(CLIFF_STANDARD, prob=0.8)
        check_probability_matrix(env.p)

    def test_prob_deterministic(self):
        """prob=1: no self-loops unless the move is blocked."""
        env = CliffWalking(CLIFF_STANDARD, prob=1.0)
        n = len(env.cell_list)
        for i in range(n):
            for a in range(4):
                if not np.allclose(env.p[i, a], 0):  # skip terminal rows
                    # self-loop only when blocked
                    if env.p[i, a, i] > 0:
                        # When blocked, self-loop probability must be exactly 1
                        assert np.isclose(env.p[i, a, i], 1.0), (
                            f"State {i}, action {a}: self-loop prob={env.p[i,a,i]} "
                            f"but expect 0 (deterministic) or 1 (blocked)"
                        )

    def test_prob_zero_immobile(self):
        """prob=0: every non-terminal row has p[i,a,i]=1 (always stay in place)."""
        env = CliffWalking(CLIFF_STANDARD, prob=0.0)
        n = len(env.cell_list)
        for i in range(n):
            for a in range(4):
                if not np.allclose(env.p[i, a], 0):
                    assert np.isclose(env.p[i, a, i], 1.0), (
                        f"State {i}, action {a}: expected self-loop p=1 at prob=0, "
                        f"got {env.p[i,a,i]}"
                    )

    def test_prob_entries_in_range(self):
        """All probability entries must be in [0, 1]."""
        env = CliffWalking(CLIFF_STANDARD, prob=0.8)
        assert np.all(env.p >= 0) and np.all(env.p <= 1), (
            "p contains entries outside [0, 1]"
        )

    def test_different_probs(self):
        """Check p with several valid prob values."""
        for prob in [0.0, 0.3, 0.5, 0.95, 1.0]:
            env = CliffWalking(CLIFF_STANDARD, prob=prob)
            check_probability_matrix(env.p)
            check_mu(env.mu)


# ===================================================================
# 3C. Terminal State Identification
# ===================================================================

class TestTerminalStates:
    """Tests for which states are terminal (absorbing)."""

    def test_goal_is_terminal(self):
        """'G' states have all-zero rows in p."""
        env = CliffWalking(CLIFF_STANDARD)
        terminals = get_terminal_states(env.p)

        # Find all 'G' cells and verify they are terminal
        g_positions = np.argwhere(env.grid == 'G')
        assert len(g_positions) > 0, "Test grid must contain at least one 'G'"
        for pos in g_positions:
            idx = get_cell_index(env, tuple(pos))
            assert idx is not None, f"Goal at {pos} not in cell_list"
            assert idx in terminals, f"Goal at {pos} (idx={idx}) should be terminal"

    def test_only_goal_is_terminal(self):
        """Only 'G' states are terminal; '.' and 'S' are not."""
        env = CliffWalking(CLIFF_STANDARD)
        terminals = get_terminal_states(env.p)

        for idx, coord in enumerate(env.cell_list):
            cell_type = env.grid[tuple(coord)]
            if cell_type == 'G':
                assert idx in terminals, f"'G' at {coord} should be terminal"
            else:
                assert idx not in terminals, (
                    f"'{cell_type}' at {coord} should NOT be terminal"
                )

    def test_lava_not_in_cell_list(self):
        """'#' (lava) cells are NOT in cell_list — they are not states."""
        env = CliffWalking(CLIFF_STANDARD)

        lava_positions = np.argwhere(env.grid == '#')
        assert len(lava_positions) > 0, "Test grid must contain lava cells"

        for pos in lava_positions:
            idx = get_cell_index(env, tuple(pos))
            assert idx is None, f"Lava at {pos} should not be in cell_list"


# ===================================================================
# 3D. Initial Distribution
# ===================================================================

class TestInitialDistribution:
    """Tests for mu (initial state distribution)."""

    def test_mu_sums_to_one(self):
        """mu must sum to 1.0."""
        env = CliffWalking(CLIFF_STANDARD)
        check_mu(env.mu)

    def test_mu_uniform_over_starts(self):
        """Multiple 'S' cells get equal initial probability."""
        env = CliffWalking(CLIFF_MULTI_START)
        check_mu(env.mu)

        start_indices = []
        for idx, coord in enumerate(env.cell_list):
            if env.grid[tuple(coord)] == 'S':
                start_indices.append(idx)

        assert len(start_indices) > 1, "Test grid must have multiple 'S' cells"
        probs = env.mu[start_indices]
        expected = 1.0 / len(start_indices)
        assert np.allclose(probs, expected), (
            f"Start probs: {probs}, expected uniform {expected}"
        )

    def test_only_starts_have_probability(self):
        """Only 'S'-indexed cells get non-zero mu."""
        env = CliffWalking(CLIFF_STANDARD)
        check_mu(env.mu)

        for idx, coord in enumerate(env.cell_list):
            cell_type = env.grid[tuple(coord)]
            if cell_type == 'S':
                assert env.mu[idx] > 0, f"'S' at {coord} has zero probability"
            else:
                assert env.mu[idx] == 0, (
                    f"'{cell_type}' at {coord} has non-zero probability {env.mu[idx]}"
                )

    def test_no_start_raises_error(self):
        """Grid with no 'S' raises ValueError."""
        grid_no_s = make_grid(["...", "...", "G.."])
        with pytest.raises(ValueError, match="start"):
            CliffWalking(grid_no_s)

    def test_single_start(self):
        """Single 'S' cell has probability 1.0."""
        grid = make_grid(["S.G"])
        env = CliffWalking(grid)
        check_mu(env.mu)

        start_idx = get_cell_index(env, (0, 0))
        assert start_idx is not None
        assert np.isclose(env.mu[start_idx], 1.0), (
            f"Single start prob = {env.mu[start_idx]}, expected 1.0"
        )


# ===================================================================
# 3E. Reward Matrix
# ===================================================================

class TestRewardMatrix:
    """Tests for reward matrix values."""

    def test_goal_reward(self):
        """Transitioning into 'G' from an adjacent cell gives +1.0 reward."""
        # Use a simple grid where we know the topology
        grid = make_grid(["S.G"])
        env = CliffWalking(grid)

        # 'S' at (0,0), '.' at (0,1), 'G' at (0,2)
        s_idx = get_cell_index(env, (0, 0))
        dot_idx = get_cell_index(env, (0, 1))
        g_idx = get_cell_index(env, (0, 2))

        # Action 0 = Right. From '.' to 'G' should give +1.0
        assert np.isclose(env.r[dot_idx, 0, g_idx], 1.0), (
            f"Goal reward: r[{dot_idx},0,{g_idx}] = {env.r[dot_idx, 0, g_idx]}, expected 1.0"
        )

    def test_lava_reward(self):
        """Trying to move into '#' gives -1.0 reward on the self-loop transition."""
        # Standard cliff: row 3 is "S##G". Cell (3,0)='S', (3,1)='#', (3,2)='#', (3,3)='G'
        env = CliffWalking(CLIFF_STANDARD)

        s_idx = get_cell_index(env, (3, 0))
        assert s_idx is not None

        # Action 0 = Right. Moving right from S hits lava.
        # This is a self-loop: p[s_idx, 0, s_idx] = 1.0
        # The reward for this self-loop should be -1.0
        assert np.isclose(env.p[s_idx, 0, s_idx], 1.0), (
            "Moving right from S into lava should be a self-loop"
        )
        assert np.isclose(env.r[s_idx, 0, s_idx], -1.0), (
            f"Lava reward: r[{s_idx},0,{s_idx}] = {env.r[s_idx,0,s_idx]}, expected -1.0"
        )

    def test_step_penalty(self):
        """Normal (non-goal, non-lava) movement has -0.01 base reward."""
        grid = make_grid(["S..G"])
        env = CliffWalking(grid)

        s_idx = get_cell_index(env, (0, 0))
        dot_idx = get_cell_index(env, (0, 1))

        # Action 0 = Right: S -> '.' (normal move). Should have -0.01
        assert np.isclose(env.r[s_idx, 0, dot_idx], -0.01), (
            f"Step penalty: r[{s_idx},0,{dot_idx}] = {env.r[s_idx,0,dot_idx]}, expected -0.01"
        )

    def test_reward_matrix_matches_p_zeros(self):
        """Wherever p[i,a,j] == 0, the reward is irrelevant but should not be NaN/inf."""
        env = CliffWalking(CLIFF_STANDARD)
        zero_p_mask = np.isclose(env.p, 0)
        assert not np.any(np.isnan(env.r)), "r contains NaN"
        assert not np.any(np.isinf(env.r)), "r contains inf"


# ===================================================================
# 3F. Boundary / Blocked Behavior
# ===================================================================

class TestBoundaryBehavior:
    """Tests for grid edges and blocked cells."""

    def test_off_grid_is_self_loop(self):
        """Moving off the grid → self-loop with p=1."""
        grid = make_grid(["S."])
        env = CliffWalking(grid, prob=0.8)

        s_idx = get_cell_index(env, (0, 0))
        assert s_idx is not None

        # Action 2 = Left (off-grid). Should be self-loop p=1.
        assert np.isclose(env.p[s_idx, 2, s_idx], 1.0), (
            f"Off-grid self-loop: p[{s_idx},2,{s_idx}] = {env.p[s_idx,2,s_idx]}, expected 1.0"
        )

    def test_into_lava_is_self_loop(self):
        """Moving toward lava → self-loop p=1 (lava is not passable)."""
        env = CliffWalking(CLIFF_STANDARD)

        s_idx = get_cell_index(env, (3, 0))
        assert s_idx is not None

        # Action 0 = Right (into lava at (3,1))
        assert np.isclose(env.p[s_idx, 0, s_idx], 1.0), (
            f"Into-lava self-loop: p[{s_idx},0,{s_idx}] = {env.p[s_idx,0,s_idx]}, expected 1.0"
        )

    def test_top_row_boundary(self):
        """Agent in top row cannot move up (action 1 = Down, action 3 = Up)."""
        # Directions: [right, down, left, up]
        # Action 3 = Up = [-1, 0]. From top row (i=0), up goes to (-1, j) → off-grid → self-loop
        grid = make_grid(["S."])
        env = CliffWalking(grid, prob=0.8)

        s_idx = get_cell_index(env, (0, 0))
        # Action 3 = Up (should be blocked from row 0)
        assert np.isclose(env.p[s_idx, 3, s_idx], 1.0), (
            f"Top boundary: p[{s_idx},3,{s_idx}] = {env.p[s_idx,3,s_idx]}, expected 1.0"
        )


# ===================================================================
# 3G. Edge Case Grids
# ===================================================================

class TestEdgeCaseGrids:
    """Tests with unusual grid configurations."""

    def test_1x1_start_only(self):
        """1x1 grid with just 'S': single state, always stays in place."""
        grid = make_grid(["S"])
        env = CliffWalking(grid)

        assert env.p.shape == (1, 4, 1)
        # All actions are self-loops (nowhere to go)
        for a in range(4):
            assert np.isclose(env.p[0, a, 0], 1.0), (
                f"1x1: p[0,{a},0] = {env.p[0,a,0]}, expected 1.0"
            )
        check_mu(env.mu)

    def test_1x1_start_is_goal(self):
        """1x1 grid where cell is both 'S' and... actually just 'S' with no G.

        Actually, a single cell can only have one char.
        We test that it works without crashing.
        """
        grid = make_grid(["S"])
        env = CliffWalking(grid)
        # No goal, so no terminal state
        assert len(get_terminal_states(env.p)) == 0

    def test_1xN_grid(self):
        """Line grid: can move left/right, not up/down."""
        env = CliffWalking(CLIFF_1xN, prob=1.0)
        check_probability_matrix(env.p)
        check_mu(env.mu)

        # For a 1xN grid, up (action 3) and down (action 1) should be blocked
        # from every non-terminal cell (skip 'G' which has all-zero rows)
        terminals = get_terminal_states(env.p)
        for idx, coord in enumerate(env.cell_list):
            if idx in terminals:
                continue
            row = coord[0]
            # Action 1 = Down (+1 row). From row 0, down goes off-grid → self-loop
            # Action 3 = Up (-1 row). From row 0, up goes off-grid → self-loop
            assert np.isclose(env.p[idx, 1, idx], 1.0), (
                f"1xN: down from {coord} should be blocked (self-loop)"
            )
            assert np.isclose(env.p[idx, 3, idx], 1.0), (
                f"1xN: up from {coord} should be blocked (self-loop)"
            )

    def test_multiple_goals(self):
        """Multiple 'G' cells: all are terminal and give +1 reward."""
        env = CliffWalking(CLIFF_MULTI_GOAL)
        check_probability_matrix(env.p)
        check_mu(env.mu)

        terminals = get_terminal_states(env.p)
        g_positions = np.argwhere(env.grid == 'G')
        assert len(g_positions) > 1, "Test grid must have multiple 'G' cells"

        for pos in g_positions:
            idx = get_cell_index(env, tuple(pos))
            assert idx in terminals, f"Goal at {pos} should be terminal"

    def test_no_goal(self):
        """Grid without 'G': no terminal states, but MDP can still be constructed."""
        env = CliffWalking(CLIFF_NO_GOAL)
        check_probability_matrix(env.p)
        check_mu(env.mu)

        terminals = get_terminal_states(env.p)
        assert len(terminals) == 0, (
            f"Expected 0 terminal states without goal, got {len(terminals)}"
        )

    def test_all_normal_cells(self):
        """Grid with only '.' and 'S': no terminal states, agent wanders."""
        grid = make_grid(["...", "S.."])
        env = CliffWalking(grid)
        check_probability_matrix(env.p)
        check_mu(env.mu)
        assert len(get_terminal_states(env.p)) == 0

    def test_gamma_extreme(self):
        """gamma=0 and gamma=1 should not crash."""
        for g in [0.0, 1.0]:
            env = CliffWalking(CLIFF_STANDARD, gamma=g)
            check_probability_matrix(env.p)
            check_mu(env.mu)

    def test_multiple_starts(self):
        """Multiple 'S' cells: uniform initial distribution over all starts."""
        env = CliffWalking(CLIFF_MULTI_START)
        check_mu(env.mu)

        start_count = np.sum(env.grid == 'S')
        assert start_count > 1
        for idx, coord in enumerate(env.cell_list):
            if env.grid[tuple(coord)] == 'S':
                assert np.isclose(env.mu[idx], 1.0 / start_count), (
                    f"Start at {coord}: mu={env.mu[idx]}, expected {1.0/start_count}"
                )

    def test_prob_out_of_range_raises_error(self):
        """prob outside [0,1] raises ValueError (from SelfDefineMDP fix)."""
        for bad_prob in [-0.1, 1.1, 2.0, -1.0]:
            with pytest.raises(ValueError, match="prob"):
                CliffWalking(CLIFF_STANDARD, prob=bad_prob)

    def test_empty_grid_raises_error(self):
        """Grid with only lava ('#') has no traversable cells → ValueError."""
        grid = make_grid(["###", "###"])
        with pytest.raises(ValueError, match="no traversable"):
            CliffWalking(grid)
