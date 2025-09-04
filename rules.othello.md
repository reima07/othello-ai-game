# Othello (Reversi) — Official Rules & App UX
_Last updated: 2025-09-03_

## Board & Start
- 8×8 board.
- Colors: **Black** (moves first) and **White**.
- Initial discs: D4=White, E5=White, E4=Black, D5=Black.

## Legal Move & Flips
- Place your disc on an empty square so that in at least one of the 8 directions
  you sandwich ≥1 contiguous opponent discs and then one of your discs.
- After placing, flip all sandwiched lines in all qualifying directions.

## Turn / Pass
- If you have any legal move, you must play.
- If you have none, you **pass** (turn goes to opponent).
- If both sides have no moves, the game ends.

## End & Scoring
- Ends when the board is full or both sides cannot move.
- More discs wins; equal is a draw.

## App UX Requirements
- **Preview-before-commit**: clicking a legal square shows a preview (with flips).
  - **Confirm** commits; **Cancel** discards.
- **Undo**: undo last **ply** (1) or full **turn** (2 plies).
- **Pass**: enabled only when no legal moves exist.
- **Choose sides**: pick Black/White when starting a new game.
- **Game Over modal**: final B/W counts + winner (Black/White/Draw) + “New Game”.
- **Legal-move hints** recommended.
- **Figma assets** must be used for board & stones.
