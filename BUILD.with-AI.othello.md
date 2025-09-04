# Build Spec — Local Othello vs Strong AI (No LLMs)

## Goal
A local web app where a human battles a **very strong AI**. No LLMs.
- Backend: **Python + FastAPI**
- Frontend: vanilla HTML/JS (or React) served locally
- Uses **Figma assets** for visuals
- Features: preview/confirm, undo, pass, choose sides, game-over modal

## Repository Layout
/backend
/othello
board.py # rules & state
eval.py # features + weights (loadable JSON)
search.py # negamax + alpha-beta + TT + iterative deepening
opening.py # tiny JSON opening book
server.py # FastAPI API (CORS enabled)
weights.json # optional, learned weights
book.json # optional, opening book
/frontend
index.html # UI: uses Figma assets and calls API
/assets
board.svg
stone_black.svg
stone_white.svg
README.md

bash
코드 복사

## Local Run
```bash
# 1) Backend
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install fastapi uvicorn
uvicorn othello.server:app --reload --port 8000

# 2) Frontend (static server; any will do)
cd ../frontend
python -m http.server 5173 -d .   # serves http://localhost:5173
# open http://localhost:5173 in the browser
Frontend fetches API from http://localhost:8000.

CORS is enabled on the backend (allow all origins) for local dev.

Figma Integration (Cursor MCP)
Export from the current Figma file:

Node “ReversiBoard” → frontend/assets/board.svg

Node “Stone/Black” → frontend/assets/stone_black.svg

Node “Stone/White” → frontend/assets/stone_white.svg
SVG, transparent background, original aspect ratio.

API (JSON)
POST /new?human_is_black=true|false
Starts a new game and sets the human side. Returns State.

GET /state
Current game state.

POST /preview body: { "r": int, "c": int }
Returns { "valid": bool, "state": State } — resulting board if the move were played (no commit).

POST /move body: { "r": int, "c": int }
Commits the human move if legal. Returns State.

POST /ai_move?max_time_ms=2000
AI plays for the side to move. Returns State.

POST /undo body: { "plies": int }
Undo last N plies (1 = last move; 2 = full turn). Returns State.

POST /pass
Perform a pass if no legal moves exist. Returns State.

State schema
json
코드 복사
{
  "grid": number[8][8],         // 1=Black, -1=White, 0=empty
  "to_move": 1 | -1,
  "black": number, "white": number,
  "legal": number[8][8],        // legal targets for side to move (1/0)
  "terminal": boolean,
  "winner": 1 | -1 | 0 | null   // 1=Black wins, -1=White wins, 0=draw, null=ongoing
}
Frontend Requirements
Show legal-move hints; on click show preview with flips.

Confirm commits via /move; Cancel discards preview.

Undo 1/2, Pass, Choose Black/White, Game Over modal.

Render board & stones from exported Figma SVGs.

Acceptance Criteria
Rules correct in all directions; auto-pass & double-pass end-game.

Winner & final counts correct.

Undo restores exact prior state.

AI responds within time limit; plays strong (corners, mobility, etc.).