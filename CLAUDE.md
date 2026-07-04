# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A personal reinforcement learning study project built on top of **mushroom-rl**. Implements classic RL algorithms (DP, TD, MC) and grid-world MDP environments from scratch for learning purposes.

## Development Commands

Python 3.11 only. Uses **uv** for dependency management.

```bash
uv sync                          # install dependencies
uv run python src/reinforcement_learning/example/tdLearning.py   # run a script
uv run pytest                    # run all tests
uv run pytest src/reinforcement_learning/test/test_cliff_walking.py -v  # single file
```

No linting or type-checking is configured.

## Architecture

The package exposes all major classes through [`__init__.py`](src/reinforcement_learning/__init__.py).

### MDP Environments (`mdp/`)

All environments extend mushroom-rl's `FiniteMDP`, which backs them with explicit transition probability (`p`), reward (`r`), and initial distribution (`mu`) tensors, all accessible as `env.p` / `env.r` / `env.mu`.

```
FiniteMDP (mushroom_rl)
  ├── SelfDefineMDP (ABC)        — grid-based MDPs from 2D char arrays
  │     ├── CliffWalking          — cliff/lava env; 'S' start, '#' lava, 'G' goal
  │     └── GridWorldEnv          — holes env; 'S' start, '*' holes, 'G' goal, '#' walls
  └── ExampleGridWorldEnv         — NOT SelfDefineMDP; hand-crafted with bushes/shears
```

**`SelfDefineMDP`** ([selfDefineMDP.py](src/reinforcement_learning/mdp/selfDefineMDP.py)) is the abstract base. Subclasses must implement four methods:

| Method | Returns | Purpose |
|---|---|---|
| `_parse_cell_list(self)` | `list` of coords | Enumerate traversable cells from `self.grid` |
| `compute_probabilities(self, grid, cell_list, prob, **kwargs)` | `(n, 4, n)` ndarray | Transition probability matrix |
| `compute_rewards(self, grid, cell_list, **kwargs)` | `(n, 4, n)` ndarray | Reward matrix |
| `initialize_state_distribution(self, cell_list, grid, **kwargs)` | `(n,)` ndarray | Initial state distribution (must sum to 1) |

The 4 actions are Right (0), Down (1), Left (2), Up (3) — corresponding to `[[0,1],[1,0],[0,-1],[-1,0]]`. A state is terminal/absorbing when its entire row in `p` is zero (checked by `FiniteMDP.step()`).

**`SelfDefineMDP.__init__`** also enforces three validity checks before calling `super().__init__`:
- `prob` must be in `[0, 1]`
- `cell_list` must not be empty (grid has at least one traversable cell)
- `mu.sum()` must equal 1.0

### Agents (`agent/`)

Subclass mushroom-rl's `Agent`. Both use `Table` for Q-value storage:
- **`SARSA`** — on-policy TD control: `Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]`
- **`QLearning`** — off-policy TD control: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) − Q(s,a)]`

### Algorithms (`utils/algorithm/`)

Standalone functions operating on raw numpy matrices (not mushroom-rl agents):
- [`dynamic_Programming.py`](src/reinforcement_learning/utils/algorithm/dynamic_Programming.py) — `value_iteration()` and `policy_iteration()`. Useful as ground truth in tests.
- [`TDPolicyEvaluation.py`](src/reinforcement_learning/utils/algorithm/TDPolicyEvaluation.py) — `evaluate_policy_mc()`, `evaluate_policy_td()`, `evaluate_policy_td_lambda()`. **Note:** `policy_evaluation.py` is a near-duplicate; the package imports from `TDPolicyEvaluation.py`.

### Examples (`example/`)

Scripts that also serve as integration tests:
- `tdLearning.py` — `Exp` class orchestrates multi-run SARSA vs QLearning experiments on CliffWalking. Saves models to `models/`, data to `data/`, plots to `pictures/`.
- `gridWorldEnv_evaluation.py` — Compares MC, TD(0), TD(λ) policy evaluation on GridWorld.
- `exampleFiniteMDP.py` — Interactive WASD demo of ExampleGridWorldEnv.

## Known Bugs

1. **CliffWalking `render()` crash** — references `self.grid_map` (never set; only `self.grid` exists) and `self._viewer` (parent `FiniteMDP` stores it as `self.viewer`). Causes `AttributeError`.
2. **Cell list type inconsistency** — `CliffWalking._parse_cell_list` returns tuples `(i,j)`, `GridWorldEnv._parse_cell_list` returns lists `[i,j]`. Both work since consumers convert to `np.array`, but the inconsistency can surprise.

## Testing

Tests live in [`src/reinforcement_learning/test/`](src/reinforcement_learning/test/). No pytest config file exists; tests run with default settings.

### Test files (70 tests, all passing)

| File | Tests | Focus |
|---|---|---|
| [`test_cliff_walking.py`](src/reinforcement_learning/test/test_cliff_walking.py) | 33 | p/r/mu invariants, terminal states (only `'G'`), reward values (+1/−1/−0.01), boundary/off-grid, edge-case grids |
| [`test_grid_world_env.py`](src/reinforcement_learning/test/test_grid_world_env.py) | 37 | Same categories plus: two terminal types (`'G'` + `'*'`), `'.'`-only initial distribution, base reward=0 |
| [`conftest.py`](src/reinforcement_learning/test/conftest.py) | — | Shared helpers (`make_grid`, `check_probability_matrix`, `check_mu`, `get_terminal_states`, `get_cell_index`), standard test grids, pytest fixtures |

Both test files share a common 7-class structure: `TestMatrixShape`, `TestProbabilityInvariants`, `TestTerminalStates`, `TestInitialDistribution`, `TestRewardMatrix`, `TestBoundaryBehavior`, `TestEdgeCaseGrids`.
