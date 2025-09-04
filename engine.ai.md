# AI Engine Spec (No LLMs) — Alpha–Beta + Learnable Eval

## Overview
Default engine = **Iterative Deepening Negamax + Alpha–Beta** with:
- **Transposition Table** (Zobrist hashing)
- **Move Ordering**: TT/PV → corners → killers → history → the rest
- **Learnable Evaluation**: linear combination of features (ML-friendly)
- **Endgame Exact Search** when empties ≤ 12
- Optional tiny **Opening Book** (JSON)

Alternative (optional): **MCTS-UCT** mode behind a flag (not required by default).

## Module Interfaces

### `othello/board.py`
```py
EMPTY=0; BLACK=+1; WHITE=-1
class Move(TypedDict): r:int; c:int
class Board:
  grid: list[list[int]]
  to_move: int
  def legal_moves(self, color:int|None=None) -> list[Move]: ...
  def apply(self, m:Move, color:int|None=None) -> "Board": ...
  def terminal(self) -> bool: ...
  def count(self) -> tuple[int,int]: ...
  def as_tuple(self) -> tuple[int,...]: ...
othello/eval.py (learnable)
Features for the side color:

mobility = my_legal_moves − opp_legal_moves

corners = 25 × (my_corners − opp_corners)

corner_closeness = −12 × (my_adj_to_empty_corner − opp_adj_to_empty_corner)

frontier = − (my_frontier − opp_frontier) # fewer frontier is better

psqt = sum(PSQT[r][c] for my discs) − sum(PSQT[r][c] for opp discs)

parity = (my_disc_count − opp_disc_count)

Stage factor mid = clamp((empties − 10)/44, 0, 1) (1 early → 0 late)

Evaluation formula:

vbnet
코드 복사
score = mid * ( w_mob * mobility + w_psqt * psqt + w_front * frontier ) \
      + (1-mid) * ( w_par  * parity  + w_cor  * corners + w_cc * corner_closeness )
Default weights (good baseline; can be trained):

json
코드 복사
{
  "mobility": 140.0,
  "psqt": 10.0,
  "frontier": 40.0,
  "parity": 90.0,
  "corners": 800.0,
  "corner_closeness": 40.0
}
PSQT (classic):

코드 복사
 100 -20  10   5   5  10 -20 100
 -20 -50  -2  -2  -2  -2 -50 -20
  10  -2   0   0   0   0  -2  10
   5  -2   0   0   0   0  -2   5
   5  -2   0   0   0   0  -2   5
  10  -2   0   0   0   0  -2  10
 -20 -50  -2  -2  -2  -2 -50 -20
 100 -20  10   5   5  10 -20 100
othello/search.py
Zobrist hashing: 64 squares × 3 piece states (empty, black, white) + side-to-move bit.

TT entry: {depth, value, flag, best} where flag ∈ {EXACT, LOWER, UPPER}.

Killer/history heuristics to boost cutoffs.

Endgame: if empties ≤ 12, search to terminal with exact disc-diff evaluation.

Iterative deepening loop:

py
코드 복사
def search_best_move(board, color, max_time_ms=2000, max_depth=64):
    start = now()
    for depth in 1..max_depth:
        (score, best) = negamax(board, color, depth, -INF, +INF, start, max_time_ms)
        save principal variation; break if time exceeded
    return best, score, nodes
Negamax with Alpha–Beta & TT (pseudocode):

csharp
코드 복사
def negamax(b, color, depth, alpha, beta):
    if time over: raise Timeout
    if TT has (b) with depth' >= depth: use bound
    if depth == 0 or b.terminal(): return evaluate(b, color), None
    moves = legal_moves(b, color)
    if moves empty:
        # pass
        return -negamax(b (to_move swapped), -color, depth-1, -beta, -alpha)
    order moves:
        [TT move] → [corner moves] → [killer moves] → [history] → others
    best=None
    for m in moves:
        b2 = apply(b, m, color)
        score = -negamax(b2, -color, depth-1, -beta, -alpha)
        if score > alpha: alpha = score; best = m
        if alpha >= beta: update killers/history; break
    store TT with flag (EXACT/LOWER/UPPER) and best
    return alpha, best
othello/opening.py
book.json format: { "hash_hex": [r, c], ... } where hash_hex is Zobrist of position.

Engine checks book before search.

Time Control
Default max_time_ms = 2000 per AI move; configurable via API.

Training — TD(λ) Self-Play (No LLM)
Provide train_td.py that:

Plays self-play games using the current engine at shallow depth (or ε-greedy randomization).

For each transition, computes features and updates weights with TD(λ):

vbnet
코드 복사
v(s) = w · f(s)
δ = r + γ v(s') − v(s)   (γ=1 here; terminal r ∈ {+1,0,−1})
e ← λ e + f(s)           (eligibility trace)
w ← w + α δ e
Saves learned weights to backend/weights.json.

On startup, eval.py loads weights.json if present; else uses defaults.

Suggested Params: α=1e−4, λ=0.7, ε=0.15, 200–5,000 episodes (increase for strength).

Optional: MCTS-UCT Mode
UCT: select child with Q + c * sqrt( ln(N) / (n+1) ), c≈1.4.

Playout policy: greedy by mobility/corner heuristics.

Stop by time budget, pick most-visited child.

Expose engine flag engine=mcts via API.

Testing Checklist
Unit tests for flipping logic (all 8 directions), pass, double-pass end-game.

Search returns corner immediately if available on first ply.

Deterministic with fixed seed.

Endgame exactness: with ≤12 empties, result equals disc-difference perfect value.

Strength Tips
Keep TT large enough (e.g., 64–256 MB if possible).

Aspiration windows can speed deep searches but add complexity.

Expand the opening book with top lines to boost practical Elo.