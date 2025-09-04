import json
from typing import Optional, Dict, List
from .board import Board, Move

class OpeningBook:
    def __init__(self, book_file: str = "book.json"):
        self.book_file = book_file
        self.book = self._load_book()
    
    def _load_book(self) -> Dict[str, List[int]]:
        """Load opening book from JSON file"""
        try:
            with open(self.book_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return empty book if file doesn't exist
            return {}
    
    def save_book(self):
        """Save current book to file"""
        with open(self.book_file, 'w') as f:
            json.dump(self.book, f, indent=2)
    
    def get_move(self, board: Board) -> Optional[Move]:
        """Get move from opening book for current position"""
        # Simple hash of board position
        position_hash = self._hash_position(board)
        
        if position_hash in self.book:
            move_coords = self.book[position_hash]
            return {"r": move_coords[0], "c": move_coords[1]}
        
        return None
    
    def add_move(self, board: Board, move: Move):
        """Add a move to the opening book"""
        position_hash = self._hash_position(board)
        self.book[position_hash] = [move["r"], move["c"]]
    
    def _hash_position(self, board: Board) -> str:
        """Create a simple hash of the board position"""
        # Convert board to string representation
        board_str = ""
        for row in board.grid:
            for cell in row:
                if cell == 1:  # Black
                    board_str += "B"
                elif cell == -1:  # White
                    board_str += "W"
                else:  # Empty
                    board_str += "."
        
        # Add side to move
        board_str += "B" if board.to_move == 1 else "W"
        
        return board_str
    
    def get_book_size(self) -> int:
        """Get number of positions in the book"""
        return len(self.book)
    
    def clear_book(self):
        """Clear all entries from the book"""
        self.book.clear()
