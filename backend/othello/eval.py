import json
from typing import List, Tuple
from .board import Board, BLACK, WHITE

# PSQT (Position Square Table) - classic Othello evaluation
PSQT = [
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [ 10,  -2,   0,   0,   0,   0,  -2,  10],
    [  5,  -2,   0,   0,   0,   0,  -2,   5],
    [  5,  -2,   0,   0,   0,   0,  -2,   5],
    [ 10,  -2,   0,   0,   0,   0,  -2,  10],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [100, -20,  10,   5,   5,  10, -20, 100]
]

# Corner positions
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

# Adjacent to corner positions
ADJACENT_TO_CORNERS = [
    (0, 1), (1, 0), (1, 1),  # Adjacent to (0, 0)
    (0, 6), (1, 6), (1, 7),  # Adjacent to (0, 7)
    (6, 0), (6, 1), (7, 1),  # Adjacent to (7, 0)
    (6, 6), (6, 7), (7, 6)   # Adjacent to (7, 7)
]

class Evaluator:
    def __init__(self, weights_file: str = "weights.json"):
        """Initialize evaluator with weights from file or defaults"""
        self.weights = self._load_weights(weights_file)
    
    def _load_weights(self, weights_file: str) -> dict:
        """Load weights from file or use defaults"""
        try:
            with open(weights_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default weights (good baseline)
            return {
                "mobility": 140.0,
                "psqt": 10.0,
                "frontier": 40.0,
                "parity": 90.0,
                "corners": 800.0,
                "corner_closeness": 40.0
            }
    
    def save_weights(self, weights_file: str = "weights.json"):
        """Save current weights to file"""
        with open(weights_file, 'w') as f:
            json.dump(self.weights, f, indent=2)
    
    def evaluate(self, board: Board, color: int) -> float:
        """Evaluate position for the given color (positive = good for color)"""
        if board.terminal():
            winner = board.winner()
            if winner == color:
                return 10000  # Win
            elif winner == -color:
                return -10000  # Loss
            else:
                return 0  # Draw
        
        # Get features for both sides
        my_features = self._get_features(board, color)
        opp_features = self._get_features(board, -color)
        
        # Calculate stage factor (mid-game vs end-game)
        empties = 64 - board.count()[0] - board.count()[1]
        mid = max(0, min(1, (empties - 10) / 44))  # 1 = early game, 0 = late game
        
        # Mid-game evaluation (mobility, PSQT, frontier)
        mid_score = (
            self.weights["mobility"] * (my_features["mobility"] - opp_features["mobility"]) +
            self.weights["psqt"] * (my_features["psqt"] - opp_features["psqt"]) +
            self.weights["frontier"] * (my_features["frontier"] - opp_features["frontier"])
        )
        
        # End-game evaluation (parity, corners, corner closeness)
        end_score = (
            self.weights["parity"] * (my_features["parity"] - opp_features["parity"]) +
            self.weights["corners"] * (my_features["corners"] - opp_features["corners"]) +
            self.weights["corner_closeness"] * (my_features["corner_closeness"] - opp_features["corner_closeness"])
        )
        
        return mid * mid_score + (1 - mid) * end_score
    
    def _get_features(self, board: Board, color: int) -> dict:
        """Calculate all features for the given color"""
        features = {}
        
        # Mobility
        features["mobility"] = len(board.legal_moves(color))
        
        # PSQT
        features["psqt"] = self._calculate_psqt(board, color)
        
        # Frontier
        features["frontier"] = self._calculate_frontier(board, color)
        
        # Parity
        features["parity"] = self._calculate_parity(board, color)
        
        # Corners
        features["corners"] = self._calculate_corners(board, color)
        
        # Corner closeness
        features["corner_closeness"] = self._calculate_corner_closeness(board, color)
        
        return features
    
    def _calculate_psqt(self, board: Board, color: int) -> float:
        """Calculate PSQT score for the given color"""
        score = 0
        for r in range(8):
            for c in range(8):
                if board.grid[r][c] == color:
                    score += PSQT[r][c]
        return score
    
    def _calculate_frontier(self, board: Board, color: int) -> int:
        """Calculate frontier discs for the given color"""
        frontier = 0
        for r in range(8):
            for c in range(8):
                if board.grid[r][c] == color and self._is_frontier(board, r, c):
                    frontier += 1
        return frontier
    
    def _is_frontier(self, board: Board, r: int, c: int) -> bool:
        """Check if a disc is on the frontier (adjacent to empty squares)"""
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and board.grid[nr][nc] == 0:
                    return True
        return False
    
    def _calculate_parity(self, board: Board, color: int) -> int:
        """Calculate disc count difference for the given color"""
        black_count, white_count = board.count()
        if color == BLACK:
            return black_count - white_count
        else:
            return white_count - black_count
    
    def _calculate_corners(self, board: Board, color: int) -> int:
        """Calculate corner count for the given color"""
        corners = 0
        for r, c in CORNERS:
            if board.grid[r][c] == color:
                corners += 1
        return corners
    
    def _calculate_corner_closeness(self, board: Board, color: int) -> int:
        """Calculate discs adjacent to empty corners for the given color"""
        closeness = 0
        for r, c in ADJACENT_TO_CORNERS:
            if board.grid[r][c] == color:
                # Check if the adjacent corner is empty
                if r == 0 or r == 7:
                    corner_r = r
                else:
                    corner_r = 0 if c == 0 or c == 1 else 7
                
                if c == 0 or c == 7:
                    corner_c = c
                else:
                    corner_c = 0 if r == 0 or r == 1 else 7
                
                if board.grid[corner_r][corner_c] == 0:
                    closeness += 1
        
        return closeness
    
    def update_weights(self, updates: dict):
        """Update weights with new values"""
        for key, value in updates.items():
            if key in self.weights:
                self.weights[key] = value
