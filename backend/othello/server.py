from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json

from .board import Board, Move, BLACK, WHITE
import math
from .eval import Evaluator
from .search import SearchEngine, ENDGAME_THRESHOLD
from .opening import OpeningBook

app = FastAPI(title="Othello AI Engine")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game state
game_board = Board()
human_is_black = True
move_history = []
preview_board = None

# Initialize AI components
evaluator = Evaluator()
opening_book = OpeningBook()
search_engine = SearchEngine(evaluator, book=opening_book)

class MoveRequest(BaseModel):
    r: int
    c: int

class UndoRequest(BaseModel):
    plies: int

class GameState(BaseModel):
    grid: List[List[int]]
    to_move: int
    black: int
    white: int
    legal: List[List[int]]
    terminal: bool
    winner: Optional[int]
    last_move: Optional[Move] | None = None

class PreviewResponse(BaseModel):
    valid: bool
    state: Optional[GameState]

def _move_to_dict(m: MoveRequest) -> Move:
    """Convert Pydantic MoveRequest to internal Move dict"""
    return {"r": int(m.r), "c": int(m.c)}

def _last_played_move() -> Optional[Move]:
    """Return the most recent non-pass move from history."""
    for mv, _ in reversed(move_history):
        if mv is not None:
            return mv
    return None

def board_to_state(board: Board, last_move_override: Optional[Move] = None) -> GameState:
    """Convert Board to GameState"""
    black_count, white_count = board.count()
    last_mv = last_move_override if last_move_override else _last_played_move()
    return GameState(
        grid=board.grid,
        to_move=board.to_move,
        black=black_count,
        white=white_count,
        legal=board.get_legal_grid(),
        terminal=board.terminal(),
        winner=board.winner(),
        last_move=last_mv
    )

@app.post("/reload_weights")
async def reload_weights():
    """Reload evaluation weights from weights.json without restarting.
    Sanitizes NaN/Inf to keep JSON and engine stable.
    """
    try:
        new_w = evaluator._load_weights("weights.json")
        # Sanitize numbers
        for k, v in list(new_w.items()):
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                new_w[k] = 0.0
        evaluator.weights = new_w
        # Reset search heuristics (optional)
        search_engine.tt.table.clear()
        return {"reloaded": True, "weights": evaluator.weights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/new")
async def new_game(human_black: bool = True):
    """Start a new game"""
    global game_board, move_history, preview_board, human_is_black
    
    game_board = Board()
    # Remember the chosen side so other endpoints (or future features) can use it
    human_is_black = human_black
    move_history = []
    preview_board = None
    
    return board_to_state(game_board)

@app.get("/state")
async def get_state():
    """Get current game state"""
    return board_to_state(game_board)

@app.post("/preview", response_model=PreviewResponse)
async def preview_move(move: MoveRequest):
    """Preview a move without committing it"""
    global preview_board
    
    try:
        # Check if move is legal using public API
        legal = game_board.legal_moves()
        if not any(m["r"] == move.r and m["c"] == move.c for m in legal):
            return PreviewResponse(valid=False, state=None)
        
        # Create preview board
        move_dict = _move_to_dict(move)
        preview_board = game_board.apply(move_dict)
        
        return PreviewResponse(valid=True, state=board_to_state(preview_board, last_move_override=move_dict))
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/move")
async def make_move(move: MoveRequest):
    """Commit a human move"""
    global game_board, move_history, preview_board
    
    try:
        # Check if move is legal using public API for consistency
        legal = game_board.legal_moves()
        if not any(m["r"] == move.r and m["c"] == move.c for m in legal):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid move",
                    "requested": {"r": move.r, "c": move.c},
                    "to_move": game_board.to_move,
                    "legal": legal,
                },
            )
        
        # Apply move
        move_dict = _move_to_dict(move)
        color_played = game_board.to_move
        game_board = game_board.apply(move_dict)
        # Record the move with the color that actually played
        move_history.append((move_dict, color_played))
        preview_board = None
        
        return board_to_state(game_board)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ai_move")
async def ai_move(max_time_ms: int = 2000):
    """AI plays for the side to move"""
    global game_board, move_history
    
    try:
        # Check if game is over
        if game_board.terminal():
            raise HTTPException(status_code=400, detail="Game is over")
        
        # Check if current side has legal moves
        if not game_board.legal_moves():
            # Pass
            color_passed = game_board.to_move
            game_board.to_move = -game_board.to_move
            move_history.append((None, color_passed))
            return board_to_state(game_board)
        
        # AI search
        ai_color = game_board.to_move
        best_move, score, nodes = search_engine.search_best_move(
            game_board, ai_color, max_time_ms
        )
        
        if best_move:
            game_board = game_board.apply(best_move)
            move_history.append((best_move, ai_color))
        else:
            # Pass if no move found
            game_board.to_move = -game_board.to_move
            move_history.append((None, ai_color))
        
        return board_to_state(game_board)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/undo")
async def undo_moves(request: UndoRequest):
    """Undo last N plies"""
    global game_board, move_history
    
    plies = request.plies
    
    if plies <= 0 or plies > len(move_history):
        raise HTTPException(status_code=400, detail="Invalid number of plies")
    
    # Reconstruct board from history
    game_board = Board()
    move_history = move_history[:-plies]
    
    for move, color_played in move_history:
        if move is not None:  # Normal move by color_played
            game_board = game_board.apply(move, color_played)
        else:  # Pass by color_played
            game_board.to_move = -color_played
    
    return board_to_state(game_board)

@app.post("/pass")
async def pass_move():
    """Perform a pass if no legal moves exist"""
    global game_board, move_history
    
    if game_board.legal_moves():
        raise HTTPException(status_code=400, detail="Legal moves available")
    
    # Pass
    game_board.to_move = -game_board.to_move
    move_history.append((None, game_board.to_move))
    
    return board_to_state(game_board)

@app.get("/info")
async def get_info():
    """Get engine information"""
    return {
        "engine": "Alpha-Beta + Iterative Deepening",
        "evaluation": "Learnable Features",
        "transposition_table": True,
        "move_ordering": True,
        "opening_book_size": opening_book.get_book_size(),
        "endgame_threshold": ENDGAME_THRESHOLD
    }

# Opening book management
class BookMove(BaseModel):
    r: int
    c: int

@app.get("/book/size")
async def book_size():
    return {"size": opening_book.get_book_size()}

@app.get("/book/move")
async def book_move():
    move = opening_book.get_move(game_board)
    return {"move": move}

@app.post("/book/add")
async def book_add(move: BookMove):
    m: Move = {"r": int(move.r), "c": int(move.c)}
    opening_book.add_move(game_board, m)
    opening_book.save_book()
    return {"added": True, "size": opening_book.get_book_size(), "move": m}

@app.post("/book/clear")
async def book_clear():
    opening_book.clear_book()
    opening_book.save_book()
    return {"cleared": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
