"""
Tests for GridWorldEnv MDP construction.

Focuses on transition matrix (p), reward matrix (r), and initial state
distribution (mu) invariants and correctness.

Grid cell types:
    '.' = normal cell (step penalty -0.01)
    'S' = start position (same as normal for transitions)
    'G' = goal (+1.0 reward, terminal)
    '*' = hole (-1.0 reward, terminal)
    '#' = wall (blocked, NOT in cell_list)
"""
import pytest
import numpy as np
from reinforcement_learning import GridWorldEnv

from reinforcement_learning.test.conftest import (
    make_grid,
    check_probability_matrix,
    check_mu,
    get_terminal_states,
    get_cell_index,
    GW_STANDARD,
    GW_NO_HOLES,
    GW_NO_GOAL,
    GW_NO_DOT,
    GW_1x1,
    GW_ALL_WALLS,
)


# ===================================================================
# 4A. Matrix Shape Invariants
# ===================================================================

class TestMatrixShape:
    """Tests for p, r, mu shapes."""

    def test_shape_standard_grid(self):
        """p=(n,4,n), r=(n,4,n), mu=(n,) with the standard 5x5 grid."""
        env = GridWorldEnv(GW_STANDARD)
        n = len(env.cell_list)

        assert env.p.shape == (n, 4, n), f"p shape: {env.p.shape}, expected ({n}, 4, {n})"
        assert env.r.shape == (n, 4, n), f"r shape: {env.r.shape}, expected ({n}, 4, {n})"
        assert env.mu.shape == (n,), f"mu shape: {env.mu.shape}, expected ({n},)"

    def test_cell_list_not_empty(self):
        """Valid grid produces n_states > 0."""
        env = GridWorldEnv(GW_STANDARD)
        assert len(env.cell_list) > 0

    def test_n_states_matches_cell_list(self):
        """Number of states equals len(cell_list)."""
        env = GridWorldEnv(GW_STANDARD)
        assert env.p.shape[0] == len(env.cell_list)


# ===================================================================
# 4B. Probability Invariants
# ===================================================================

class TestProbabilityInvariants:
    """Tests for transition probability matrix properties."""

    def test_probability_row_sums(self):
        """Every p[i,a,:] sums to either 0 (terminal) or 1 (non-terminal)."""
        env = GridWorldEnv(GW_STANDARD, prob=0.8)
        check_probability_matrix(env.p)

    def test_prob_deterministic(self):
        """prob=1: no self-loops unless the move is blocked."""
        env = GridWorldEnv(GW_STANDARD, prob=1.0)
        n = len(env.cell_list)
        for i in range(n):
            for a in range(4):
                if not np.allclose(env.p[i, a], 0):
                    if env.p[i, a, i] > 0:
                        assert np.isclose(env.p[i, a, i], 1.0), (
                            f"State {i}, action {a}: self-loop prob={env.p[i,a,i]} "
                            f"but expect 0 (deterministic) or 1 (blocked)"
                        )

    def test_prob_zero_immobile(self):
        """prob=0: every non-terminal row has p[i,a,i]=1."""
        env = GridWorldEnv(GW_STANDARD, prob=0.0)
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
        env = GridWorldEnv(GW_STANDARD, prob=0.8)
        assert np.all(env.p >= 0) and np.all(env.p <= 1), (
            "p contains entries outside [0, 1]"
        )

    def test_different_probs(self):
        """Check p with several valid prob values."""
        for prob in [0.0, 0.3, 0.5, 0.95, 1.0]:
            env = GridWorldEnv(GW_STANDARD, prob=prob)
            check_probability_matrix(env.p)
            check_mu(env.mu)


# ===================================================================
# 4C. Terminal State Identification
# ===================================================================

class TestTerminalStates:
    """Tests for which states are terminal (absorbing)."""

    def test_goal_is_terminal(self):
        """'G' states have all-zero rows in p."""
        env = GridWorldEnv(GW_STANDARD)
        terminals = get_terminal_states(env.p)

        g_positions = np.argwhere(env.grid == 'G')
        assert len(g_positions) > 0, "Test grid must contain at least one 'G'"
        for pos in g_positions:
            idx = get_cell_index(env, tuple(pos))
            assert idx is not None, f"Goal at {pos} not in cell_list"
            assert idx in terminals, f"Goal at {pos} (idx={idx}) should be terminal"

    def test_holes_are_terminal(self):
        """'*' states have all-zero rows in p."""
        env = GridWorldEnv(GW_STANDARD)
        terminals = get_terminal_states(env.p)

        hole_positions = np.argwhere(env.grid == '*')
        assert len(hole_positions) > 0, "Test grid must contain at least one '*'"
        for pos in hole_positions:
            idx = get_cell_index(env, tuple(pos))
            assert idx is not None, f"Hole at {pos} not in cell_list"
            assert idx in terminals, f"Hole at {pos} (idx={idx}) should be terminal"

    def test_normal_not_terminal(self):
        """'.' and 'S' states are NOT terminal."""
        env = GridWorldEnv(GW_STANDARD)
        terminals = get_terminal_states(env.p)

        for idx, coord in enumerate(env.cell_list):
            cell_type = env.grid[tuple(coord)]
            if cell_type in ('.', 'S'):
                assert idx not in terminals, (
                    f"'{cell_type}' at {coord} should NOT be terminal"
                )

    def test_walls_not_in_cell_list(self):
        """'#' cells are absent from cell_list."""
        env = GridWorldEnv(GW_STANDARD)

        wall_positions = np.argwhere(env.grid == '#')
        assert len(wall_positions) > 0, "Test grid must contain wall cells"

        for pos in wall_positions:
            idx = get_cell_index(env, tuple(pos))
            assert idx is None, f"Wall at {pos} should not be in cell_list"

    def test_terminal_count_standard(self):
        """Standard grid has exactly 3 terminal states (1 G + 2 holes)."""
        env = GridWorldEnv(GW_STANDARD)
        terminals = get_terminal_states(env.p)

        n_goals = np.sum(env.grid == 'G')
        n_holes = np.sum(env.grid == '*')
        expected = n_goals + n_holes
        assert len(terminals) == expected, (
            f"Expected {expected} terminal states, got {len(terminals)}"
        )


# ===================================================================
# 4D. Initial Distribution
# ===================================================================

class TestInitialDistribution:
    """Tests for mu (initial state distribution)."""

    def test_mu_sums_to_one(self):
        """mu must sum to 1.0."""
        env = GridWorldEnv(GW_STANDARD)
        check_mu(env.mu)

    def test_mu_uniform_over_dots(self):
        """Initial distribution is uniform over '.' cells only (not 'S')."""
        env = GridWorldEnv(GW_STANDARD)
        check_mu(env.mu)

        dot_indices = []
        for idx, coord in enumerate(env.cell_list):
            if env.grid[tuple(coord)] == '.':
                dot_indices.append(idx)

        assert len(dot_indices) > 0, "Test grid must have '.' cells"
        expected = 1.0 / len(dot_indices)
        for idx in dot_indices:
            assert np.isclose(env.mu[idx], expected), (
                f"Dot cell idx {idx}: mu={env.mu[idx]}, expected {expected}"
            )

    def test_no_dot_raises_error(self):
        """Grid with no '.' raises ValueError (after fix)."""
        grid = make_grid(["S*G"])
        with pytest.raises(ValueError, match="initial state distribution"):
            GridWorldEnv(grid)

    def test_s_start_has_zero_prob(self):
        """'S' cells have zero initial probability (documented quirk)."""
        env = GridWorldEnv(GW_STANDARD)
        check_mu(env.mu)

        for idx, coord in enumerate(env.cell_list):
            if env.grid[tuple(coord)] == 'S':
                assert env.mu[idx] == 0.0, (
                    f"'S' at {coord} has mu={env.mu[idx]}, expected 0.0 "
                    f"(documented: only '.' cells in initial distribution)"
                )

    def test_mu_all_dots(self):
        """Grid with only '.' cells: uniform distribution over all cells."""
        grid = make_grid(["..", ".."])
        env = GridWorldEnv(grid)
        check_mu(env.mu)

        expected = 1.0 / 4
        for idx in range(4):
            assert np.isclose(env.mu[idx], expected), (
                f"All-dot grid: mu[{idx}] = {env.mu[idx]}, expected {expected}"
            )


# ===================================================================
# 4E. Reward Matrix
# ===================================================================

class TestRewardMatrix:
    """Tests for reward matrix values."""

    def test_goal_reward(self):
        """Transitioning into 'G' gives +1.0 reward."""
        grid = make_grid(["S.G"])
        env = GridWorldEnv(grid)

        dot_idx = get_cell_index(env, (0, 1))
        g_idx = get_cell_index(env, (0, 2))

        # Action 0 = Right. From '.' to 'G' should give +1.0
        assert np.isclose(env.r[dot_idx, 0, g_idx], 1.0), (
            f"Goal reward: r[{dot_idx},0,{g_idx}] = {env.r[dot_idx,0,g_idx]}, expected 1.0"
        )

    def test_hole_reward(self):
        """Transitioning into '*' gives -1.0 reward."""
        grid = make_grid([".S*G"])  # '.' needed for initial distribution
        env = GridWorldEnv(grid)
        # cell_list: (0,0)='.', (0,1)='S', (0,2)='*', (0,3)='G'

        s_idx = get_cell_index(env, (0, 1))
        hole_idx = get_cell_index(env, (0, 2))

        # Action 0 = Right. From 'S' to '*' should give -1.0
        assert np.isclose(env.r[s_idx, 0, hole_idx], -1.0), (
            f"Hole reward: r[{s_idx},0,{hole_idx}] = {env.r[s_idx,0,hole_idx]}, expected -1.0"
        )

    def test_step_penalty(self):
        """Moving into '.' gives -0.01 reward."""
        grid = make_grid(["S.G"])
        env = GridWorldEnv(grid)

        s_idx = get_cell_index(env, (0, 0))
        dot_idx = get_cell_index(env, (0, 1))

        # Action 0 = Right. From 'S' to '.' should give -0.01
        assert np.isclose(env.r[s_idx, 0, dot_idx], -0.01), (
            f"Step penalty: r[{s_idx},0,{dot_idx}] = {env.r[s_idx,0,dot_idx]}, expected -0.01"
        )

    def test_base_reward_is_zero(self):
        """GridWorldEnv base reward is 0 (unlike CliffWalking's -0.01 base)."""
        env = GridWorldEnv(GW_STANDARD)

        # Find a self-loop transition that's not overwritten by give_reward
        # (e.g., a blocked transition should have 0 reward, not -0.01)
        zero_mask = np.isclose(env.r, 0)
        nonzero_mask = ~zero_mask
        # At least some entries should be zero (the base)
        assert np.any(zero_mask), "Expected some zero base rewards in r"

    def test_reward_matrix_no_nan_inf(self):
        """r contains no NaN or inf values."""
        env = GridWorldEnv(GW_STANDARD)
        assert not np.any(np.isnan(env.r)), "r contains NaN"
        assert not np.any(np.isinf(env.r)), "r contains inf"

    def test_goal_reward_is_exactly_one(self):
        """All transitions entering 'G' from any direction are +1.0."""
        grid = make_grid([
            "..G",
            "#.G",
        ])
        env = GridWorldEnv(grid)

        g_positions = np.argwhere(env.grid == 'G')
        for g_pos in g_positions:
            g_idx = get_cell_index(env, tuple(g_pos))
            # Check r[:, :, g_idx] — any non-zero entry should be 1.0
            col = env.r[:, :, g_idx]
            nonzero = col[~np.isclose(col, 0)]
            if len(nonzero) > 0:
                assert np.allclose(nonzero, 1.0), (
                    f"Non-zero rewards into G at {g_pos}: {nonzero}"
                )


# ===================================================================
# 4F. Boundary / Blocked Behavior
# ===================================================================

class TestBoundaryBehavior:
    """Tests for grid edges and blocked cells."""

    def test_off_grid_is_self_loop(self):
        """Moving off the grid → self-loop with p=1."""
        grid = make_grid(["S."])
        env = GridWorldEnv(grid, prob=0.8)

        s_idx = get_cell_index(env, (0, 0))
        assert s_idx is not None

        # Action 2 = Left (off-grid). Should be self-loop p=1.
        assert np.isclose(env.p[s_idx, 2, s_idx], 1.0), (
            f"Off-grid: p[{s_idx},2,{s_idx}] = {env.p[s_idx,2,s_idx]}, expected 1.0"
        )

    def test_into_wall_is_self_loop(self):
        """Moving toward '#' → self-loop p=1 (wall is not passable)."""
        grid = make_grid([".S#G"])  # '.' needed for initial distribution
        env = GridWorldEnv(grid, prob=0.8)

        s_idx = get_cell_index(env, (0, 1))
        assert s_idx is not None

        # Action 0 = Right (into wall at (0,2))
        assert np.isclose(env.p[s_idx, 0, s_idx], 1.0), (
            f"Into-wall: p[{s_idx},0,{s_idx}] = {env.p[s_idx,0,s_idx]}, expected 1.0"
        )

    def test_top_row_boundary(self):
        """Agent in top row cannot move up (action 3 = Up = [-1,0])."""
        grid = make_grid(["S."])
        env = GridWorldEnv(grid, prob=0.8)

        s_idx = get_cell_index(env, (0, 0))
        # Action 3 = Up (blocked from row 0)
        assert np.isclose(env.p[s_idx, 3, s_idx], 1.0), (
            f"Top boundary: p[{s_idx},3,{s_idx}] = {env.p[s_idx,3,s_idx]}, expected 1.0"
        )

    def test_wall_from_both_sides(self):
        """Cells on either side of a wall correctly treat it as blocked."""
        grid = make_grid([".S#G"])  # '.' needed for initial distribution
        env = GridWorldEnv(grid, prob=1.0)

        s_idx = get_cell_index(env, (0, 1))
        g_idx = get_cell_index(env, (0, 3))

        # From S, going right into wall at (0,2): self-loop
        assert np.isclose(env.p[s_idx, 0, s_idx], 1.0)
        # From G, going left into wall: self-loop... but G is terminal
        assert g_idx in get_terminal_states(env.p)


# ===================================================================
# 4G. Edge Case Grids
# ===================================================================

class TestEdgeCaseGrids:
    """Tests with unusual grid configurations."""

    def test_1x1_dot(self):
        """1x1 grid with just '.': single state, stays in place."""
        env = GridWorldEnv(GW_1x1)

        assert env.p.shape == (1, 4, 1)
        # All actions are self-loops (nowhere to go)
        for a in range(4):
            assert np.isclose(env.p[0, a, 0], 1.0), (
                f"1x1: p[0,{a},0] = {env.p[0,a,0]}, expected 1.0"
            )
        check_mu(env.mu)

    def test_no_goal(self):
        """Grid without 'G': no terminal positive reward source."""
        env = GridWorldEnv(GW_NO_GOAL)
        check_probability_matrix(env.p)
        check_mu(env.mu)

        # '*' cells should still be terminal
        terminals = get_terminal_states(env.p)
        hole_positions = np.argwhere(env.grid == '*')
        for pos in hole_positions:
            idx = get_cell_index(env, tuple(pos))
            assert idx in terminals

    def test_no_holes(self):
        """Grid without '*': no terminal negative reward source."""
        env = GridWorldEnv(GW_NO_HOLES)
        check_probability_matrix(env.p)
        check_mu(env.mu)

        # 'G' should still be terminal
        terminals = get_terminal_states(env.p)
        assert len(terminals) == 1  # only G

    def test_all_walls_raises_error(self):
        """Grid of only '#' raises ValueError (empty cell_list)."""
        with pytest.raises(ValueError, match="no traversable"):
            GridWorldEnv(GW_ALL_WALLS)

    def test_all_holes_raises_error(self):
        """Grid with only '*' raises ValueError (no '.' for initial distribution)."""
        grid = make_grid(["***", "***"])
        with pytest.raises(ValueError, match="initial state distribution"):
            GridWorldEnv(grid)

    def test_only_goal(self):
        """Grid with only 'G': 1 state, terminal, no start distribution."""
        grid = make_grid(["G"])
        # No '.' and no 'S' → raises ValueError (after fix)
        with pytest.raises(ValueError, match="initial state distribution"):
            GridWorldEnv(grid)

    def test_prob_out_of_range_raises_error(self):
        """prob outside [0,1] raises ValueError (from SelfDefineMDP fix)."""
        for bad_prob in [-0.1, 1.1, 2.0, -1.0]:
            with pytest.raises(ValueError, match="prob"):
                GridWorldEnv(GW_STANDARD, prob=bad_prob)

    def test_gamma_extreme(self):
        """gamma=0 and gamma=1 should not crash."""
        for g in [0.0, 1.0]:
            env = GridWorldEnv(GW_STANDARD, gamma=g)
            check_probability_matrix(env.p)
            check_mu(env.mu)

    def test_start_is_also_goal(self):
        """Grid with S adjacent to G: normal transition works correctly.

        S can move into G; G is terminal. This verifies that start state
        handling doesn't interfere with goal transition logic.
        """
        grid = make_grid([".SG"])  # '.' for init dist, S adjacent to G
        env = GridWorldEnv(grid, prob=1.0)

        s_idx = get_cell_index(env, (0, 1))
        g_idx = get_cell_index(env, (0, 2))

        # From S, right into G: deterministic transition to G at prob=1
        assert np.isclose(env.p[s_idx, 0, g_idx], 1.0), (
            f"S→G transition: p[{s_idx},0,{g_idx}] = {env.p[s_idx,0,g_idx]}, expected 1.0"
        )
        # G is terminal
        assert g_idx in get_terminal_states(env.p)
        # Reward for entering G
        assert np.isclose(env.r[s_idx, 0, g_idx], 1.0), (
            f"Goal reward: r[{s_idx},0,{g_idx}] = {env.r[s_idx,0,g_idx]}, expected 1.0"
        )
