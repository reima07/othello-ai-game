import time
import random
from typing import Tuple, Optional, List, Dict
from .board import Board, Move, BLACK, WHITE
from .eval import Evaluator
from .opening import OpeningBook

# Constants
INF = float('inf')
EXACT = 0
LOWER = 1
UPPER = 2

class TranspositionTable:
    def __init__(self, max_size: int = 1000000):
        self.table = {}
        self.max_size = max_size
    
    def _zobrist_hash(self, board: Board) -> int:
        """Calculate Zobrist hash for the board position"""
        hash_val = 0
        for r in range(8):
            for c in range(8):
                piece = board.grid[r][c]
                if piece != 0:
                    # Use a simple hash function for demo
                    hash_val ^= (piece + 1) << (r * 8 + c)
        hash_val ^= (board.to_move + 1) << 64
        return hash_val
    
    def get(self, board: Board) -> Optional[Tuple[int, float, int, Optional[Move]]]:
        """Get TT entry: (depth, value, flag, best_move)"""
        hash_val = self._zobrist_hash(board)
        return self.table.get(hash_val)
    
    def store(self, board: Board, depth: int, value: float, flag: int, best_move: Optional[Move]):
        """Store TT entry"""
        if len(self.table) >= self.max_size:
            # Simple replacement strategy: clear half the table
            keys = list(self.table.keys())
            for key in keys[:len(keys)//2]:
                del self.table[key]
        
        hash_val = self._zobrist_hash(board)
        self.table[hash_val] = (depth, value, flag, best_move)

ENDGAME_THRESHOLD = 12  # empties at which we solve exactly

class SearchEngine:
    def __init__(self, evaluator: Evaluator, book: Optional[OpeningBook] = None):
        self.evaluator = evaluator
        self.book = book
        self.tt = TranspositionTable()
        self.killers = [[None, None] for _ in range(64)]  # [depth][2 moves]
        self.history = [[0 for _ in range(8)] for _ in range(8)]  # [r][c] -> count
        self.end_tt: Dict[Tuple[int, ...], Tuple[int, Optional[Move]]] = {}
    
    def search_best_move(self, board: Board, color: int, max_time_ms: int = 2000, max_depth: int = 64) -> Tuple[Optional[Move], float, int]:
        """Find best move using iterative deepening"""
        start_time = time.time()
        best_move = None
        best_score = -INF
        nodes_searched = 0
        
        # Simple tactical rules before searching
        legal = board.legal_moves(color)
        if not legal:
            return None, 0, 0
        if len(legal) == 1:
            return legal[0], 0, 0
        # Prefer corners instantly
        for m in legal:
            if (m["r"], m["c"]) in [(0,0),(0,7),(7,0),(7,7)]:
                return m, 0, 0

        # Hard-avoid poison moves that hand a corner to the opponent, if any safe alternatives exist
        safe_moves = []
        poison_moves = []
        for m in legal:
            if self._is_poison_move(board, m, color):
                poison_moves.append(m)
            else:
                safe_moves.append(m)
        if safe_moves:
            legal = safe_moves

        # Check opening book first
        book_move = self._get_book_move(board)
        if book_move:
            return book_move, 0, 0

        # Exact endgame solve when empties are small (best-effort: fall back on timeout)
        empties = 64 - sum(board.count())
        if empties <= ENDGAME_THRESHOLD:
            try:
                score, best, nodes = self._endgame_negamax(board, color, -INF, INF, start_time, max_time_ms)
                return best, score, nodes
            except TimeoutError:
                # Fall back to normal search within time budget
                pass
        
        # Iterative deepening
        for depth in range(1, max_depth + 1):
            try:
                score, move, nodes = self._negamax(board, color, depth, -INF, INF, start_time, max_time_ms)
                nodes_searched += nodes
                
                if move:
                    best_move = move
                    best_score = score
                
                # Time check
                if time.time() - start_time > max_time_ms / 1000:
                    break
                    
            except TimeoutError:
                break
        
        return best_move, best_score, nodes_searched
    
    def _negamax(self, board: Board, color: int, depth: int, alpha: float, beta: float, 
                  start_time: float, max_time_ms: int) -> Tuple[float, Optional[Move], int]:
        """Negamax with alpha-beta pruning and TT"""
        # Time check
        if time.time() - start_time > max_time_ms / 1000:
            raise TimeoutError()
        
        # Check TT
        tt_entry = self.tt.get(board)
        if tt_entry:
            tt_depth, tt_value, tt_flag, tt_best = tt_entry
            if tt_depth >= depth:
                if tt_flag == EXACT:
                    return tt_value, tt_best, 1
                elif tt_flag == LOWER and tt_value >= beta:
                    return tt_value, tt_best, 1
                elif tt_flag == UPPER and tt_value <= alpha:
                    return tt_value, tt_best, 1
        
        # Terminal node
        if depth == 0 or board.terminal():
            eval_score = self.evaluator.evaluate(board, color)
            return eval_score, None, 1
        
        # Pass move if no legal moves
        legal_moves = board.legal_moves(color)
        if not legal_moves:
            # Pass
            new_board = board.copy()
            new_board.to_move = -color
            score, _, nodes = self._negamax(new_board, -color, depth - 1, -beta, -alpha, start_time, max_time_ms)
            return -score, None, nodes + 1
        
        # Move ordering
        ordered_moves = self._order_moves(board, legal_moves, depth, color)
        
        best_move = None
        best_score = -INF
        total_nodes = 1
        
        # Alpha-beta search
        for move in ordered_moves:
            new_board = board.apply(move, color)
            score, _, nodes = self._negamax(new_board, -color, depth - 1, -beta, -alpha, start_time, max_time_ms)
            score = -score
            total_nodes += nodes
            
            if score > best_score:
                best_score = score
                best_move = move
            
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    # Beta cutoff - update killers and history
                    self._update_killers(move, depth)
                    self._update_history(move, depth)
                    break
        
        # Store in TT
        flag = EXACT
        if best_score <= alpha:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        
        self.tt.store(board, depth, best_score, flag, best_move)
        
        return best_score, best_move, total_nodes
    
    def _order_moves(self, board: Board, moves: List[Move], depth: int, color: int) -> List[Move]:
        """Order moves for better alpha-beta pruning"""
        if not moves:
            return moves
        
        # TT move first
        tt_entry = self.tt.get(board)
        tt_best = tt_entry[3] if tt_entry else None
        
        # Corner moves
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        corner_moves = []
        other_moves = []
        
        for move in moves:
            if (move["r"], move["c"]) in corners:
                corner_moves.append(move)
            else:
                other_moves.append(move)
        
        # Killer moves
        killer_moves = []
        non_killer_moves = []
        # Depth may exceed killers array bound; clamp index
        killers_idx = min(max(depth, 0), len(self.killers) - 1)
        killers_list = [k for k in self.killers[killers_idx] if k]
        
        def is_killer(m: Move) -> bool:
            return any(k["r"] == m["r"] and k["c"] == m["c"] for k in killers_list)
        
        for move in other_moves:
            if is_killer(move):
                killer_moves.append(move)
            else:
                non_killer_moves.append(move)
        
        # Risky squares (X/C) near empty corners → push later
        x_squares = [ (1,1),(1,6),(6,1),(6,6) ]
        c_squares = [ (0,1),(1,0),(0,6),(1,7),(7,1),(6,0),(6,7),(7,6) ]
        empty_corners = {(0,0): board.grid[0][0]==0,
                         (0,7): board.grid[0][7]==0,
                         (7,0): board.grid[7][0]==0,
                         (7,7): board.grid[7][7]==0}
        def is_risky(m: Move) -> bool:
            rc = (m["r"], m["c"])
            if rc in x_squares:
                # Map X to its corner
                mapping = {(1,1):(0,0),(1,6):(0,7),(6,1):(7,0),(6,6):(7,7)}
                return empty_corners[mapping[rc]]
            if rc in c_squares:
                mapping = {(0,1):(0,0),(1,0):(0,0),(0,6):(0,7),(1,7):(0,7),
                           (7,1):(7,0),(6,0):(7,0),(6,7):(7,7),(7,6):(7,7)}
                return empty_corners[mapping[rc]]
            return False

        safe_moves = [m for m in non_killer_moves if not is_risky(m)]
        risky_moves = [m for m in non_killer_moves if is_risky(m)]

        def is_edge(m: Move) -> bool:
            r, c = m["r"], m["c"]
            return r == 0 or r == 7 or c == 0 or c == 7

        # Within safe moves, prefer edges first
        safe_edges = [m for m in safe_moves if is_edge(m)]
        safe_inners = [m for m in safe_moves if not is_edge(m)]

        # Sort by history within groups
        safe_edges.sort(key=lambda m: self.history[m["r"]][m["c"]], reverse=True)
        safe_inners.sort(key=lambda m: self.history[m["r"]][m["c"]], reverse=True)
        risky_moves.sort(key=lambda m: self.history[m["r"]][m["c"]], reverse=True)
        
        # Combine in order: TT -> corners -> killers -> history -> others
        ordered: List[Move] = []
        if tt_best and any(m["r"] == tt_best["r"] and m["c"] == tt_best["c"] for m in moves):
            ordered.append(tt_best)
        
        ordered.extend([m for m in corner_moves if not (tt_best and m == tt_best)])
        ordered.extend([m for m in killer_moves if not (tt_best and m == tt_best)])
        ordered.extend([m for m in safe_edges if not (tt_best and m == tt_best)])
        ordered.extend([m for m in safe_inners if not (tt_best and m == tt_best)])
        ordered.extend([m for m in risky_moves if not (tt_best and m == tt_best)])

        return ordered

    # --- Tactical helpers ---
    def _is_poison_move(self, board: Board, move: Move, color: int) -> bool:
        """Return True if this move likely hands a corner next ply or is an X/C next to empty corner."""
        r, c = move["r"], move["c"]
        x_squares = {(1,1):(0,0),(1,6):(0,7),(6,1):(7,0),(6,6):(7,7)}
        c_squares = {(0,1):(0,0),(1,0):(0,0),(0,6):(0,7),(1,7):(0,7),
                     (7,1):(7,0),(6,0):(7,0),(6,7):(7,7),(7,6):(7,7)}
        rc = (r, c)
        # If we play X/C while adjacent corner is empty, treat as poison
        if rc in x_squares and board.grid[x_squares[rc][0]][x_squares[rc][1]] == 0:
            return True
        if rc in c_squares and board.grid[c_squares[rc][0]][c_squares[rc][1]] == 0:
            return True
        # Simulate and see if opponent gets a corner immediately
        b2 = board.apply(move, color)
        for m in b2.legal_moves(-color):
            if (m["r"], m["c"]) in [(0,0),(0,7),(7,0),(7,7)]:
                return True
        return False
    
    def _update_killers(self, move: Move, depth: int):
        """Update killer moves"""
        if depth < len(self.killers):
            killers = self.killers[depth]
            if move not in killers:
                killers[1] = killers[0]
                killers[0] = move
    
    def _update_history(self, move: Move, depth: int):
        """Update history heuristic"""
        self.history[move["r"]][move["c"]] += depth * depth
    
    def _get_book_move(self, board: Board) -> Optional[Move]:
        """Get move from opening book if available"""
        if self.book:
            return self.book.get_move(board)
        return None

    def _disc_diff(self, board: Board, color: int) -> int:
        b, w = board.count()
        return (b - w) if color == BLACK else (w - b)

    def _endgame_negamax(self, board: Board, color: int, alpha: float, beta: float,
                          start_time: float, max_time_ms: int) -> Tuple[int, Optional[Move], int]:
        """Exact endgame search to terminal with disc-difference scoring."""
        # Time guard
        if time.time() - start_time > max_time_ms / 1000:
            raise TimeoutError()

        key = board.as_tuple()
        if key in self.end_tt:
            val, best = self.end_tt[key]
            return val, best, 1

        if board.terminal():
            val = self._disc_diff(board, color)
            self.end_tt[key] = (val, None)
            return val, None, 1

        moves = board.legal_moves(color)
        if not moves:
            # Pass
            b2 = board.copy()
            b2.to_move = -color
            score, _, nodes = self._endgame_negamax(b2, -color, -beta, -alpha, start_time, max_time_ms)
            score = -score
            self.end_tt[key] = (score, None)
            return score, None, nodes + 1

        best = None
        best_score = -INF
        nodes_total = 1
        for m in moves:
            b2 = board.apply(m, color)
            score, _, nodes = self._endgame_negamax(b2, -color, -beta, -alpha, start_time, max_time_ms)
            score = -score
            nodes_total += nodes
            if score > best_score:
                best_score = score
                best = m
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break

        self.end_tt[key] = (int(best_score), best)
        return int(best_score), best, nodes_total

class MCTSEngine:
    """Optional MCTS-UCT engine (not required by default)"""
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.c = 1.4  # UCT constant
    
    def search_best_move(self, board: Board, color: int, max_time_ms: int = 2000) -> Tuple[Optional[Move], float, int]:
        """MCTS search (simplified implementation)"""
        # This is a placeholder for the optional MCTS mode
        # For now, fall back to minimax
        engine = SearchEngine(self.evaluator)
        return engine.search_best_move(board, color, max_time_ms)
