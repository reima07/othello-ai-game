from typing import List, Tuple, Optional, TypedDict
import copy

# Constants
EMPTY = 0
BLACK = 1
WHITE = -1

class Move(TypedDict):
    r: int
    c: int

class Board:
    def __init__(self):
        # Initialize 8x8 board
        self.grid = [[EMPTY for _ in range(8)] for _ in range(8)]
        self.to_move = BLACK  # Black moves first
        
        # Set initial pieces (standard Othello starting position)
        # D4 (3,3) = White, E5 (4,4) = White
        # E4 (3,4) = Black, D5 (4,3) = Black
        self.grid[3][3] = WHITE  # D4
        self.grid[4][4] = WHITE  # E5
        self.grid[3][4] = BLACK  # E4
        self.grid[4][3] = BLACK  # D5
    
    def copy(self) -> 'Board':
        """Create a deep copy of the board"""
        new_board = Board()
        new_board.grid = copy.deepcopy(self.grid)
        new_board.to_move = self.to_move
        return new_board
    
    def legal_moves(self, color: Optional[int] = None) -> List[Move]:
        """Get legal moves for the given color (or current side to move)"""
        if color is None:
            color = self.to_move
        
        legal_moves = []
        for r in range(8):
            for c in range(8):
                if self.grid[r][c] == EMPTY and self._is_legal_move(r, c, color):
                    legal_moves.append({"r": r, "c": c})
        return legal_moves
    
    def _is_legal_move(self, r: int, c: int, color: int) -> bool:
        """Check if placing a piece at (r, c) is legal for the given color"""
        if self.grid[r][c] != EMPTY:
            return False
        
        # Check all 8 directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            if self._can_flip_in_direction(r, c, dr, dc, color):
                return True
        return False
    
    def _can_flip_in_direction(self, r: int, c: int, dr: int, dc: int, color: int) -> bool:
        """Check if pieces can be flipped in a specific direction"""
        opponent = -color
        r += dr
        c += dc
        
        # Must have at least one opponent piece
        if not (0 <= r < 8 and 0 <= c < 8) or self.grid[r][c] != opponent:
            return False
        
        # Continue in this direction until we find our own piece
        while 0 <= r < 8 and 0 <= c < 8:
            if self.grid[r][c] == color:
                return True
            elif self.grid[r][c] == EMPTY:
                return False
            r += dr
            c += dc
        
        return False
    
    def apply(self, move: Move, color: Optional[int] = None) -> 'Board':
        """Apply a move and return a new board"""
        if color is None:
            color = self.to_move
        
        if not self._is_legal_move(move["r"], move["c"], color):
            raise ValueError(f"Invalid move: {move}")
        
        new_board = self.copy()
        new_board._place_piece(move["r"], move["c"], color)
        new_board.to_move = -color
        return new_board
    
    def _place_piece(self, r: int, c: int, color: int):
        """Place a piece and flip opponent pieces"""
        self.grid[r][c] = color
        
        # Flip pieces in all valid directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            self._flip_in_direction(r, c, dr, dc, color)
    
    def _flip_in_direction(self, r: int, c: int, dr: int, dc: int, color: int):
        """Flip opponent pieces in a specific direction"""
        opponent = -color
        r += dr
        c += dc
        to_flip = []
        
        # Collect pieces to flip
        while 0 <= r < 8 and 0 <= c < 8:
            if self.grid[r][c] == opponent:
                to_flip.append((r, c))
            elif self.grid[r][c] == color:
                # Flip all collected pieces
                for flip_r, flip_c in to_flip:
                    self.grid[flip_r][flip_c] = color
                break
            else:  # EMPTY
                break
            r += dr
            c += dc
    
    def terminal(self) -> bool:
        """Check if the game is over"""
        if self.count()[0] + self.count()[1] == 64:  # Board is full
            return True
        
        # Check if both sides have no legal moves
        black_moves = len(self.legal_moves(BLACK))
        white_moves = len(self.legal_moves(WHITE))
        
        return black_moves == 0 and white_moves == 0
    
    def count(self) -> Tuple[int, int]:
        """Return (black_count, white_count)"""
        black_count = sum(1 for row in self.grid for cell in row if cell == BLACK)
        white_count = sum(1 for row in self.grid for cell in row if cell == WHITE)
        return black_count, white_count
    
    def as_tuple(self) -> Tuple[int, ...]:
        """Convert board to tuple for hashing"""
        result = []
        for row in self.grid:
            result.extend(row)
        result.append(self.to_move)
        return tuple(result)
    
    def get_legal_grid(self) -> List[List[int]]:
        """Get 8x8 grid showing legal moves (1 for legal, 0 for illegal)"""
        legal_grid = [[0 for _ in range(8)] for _ in range(8)]
        legal_moves = self.legal_moves()
        for move in legal_moves:
            legal_grid[move["r"]][move["c"]] = 1
        return legal_grid
    
    def winner(self) -> Optional[int]:
        """Return winner: 1 (Black), -1 (White), 0 (Draw), or None (ongoing)"""
        if not self.terminal():
            return None
        
        black_count, white_count = self.count()
        if black_count > white_count:
            return BLACK
        elif white_count > black_count:
            return WHITE
        else:
            return 0
