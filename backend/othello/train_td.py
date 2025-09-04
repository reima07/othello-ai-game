import json
import random
from typing import List, Tuple
from .board import Board, Move, BLACK, WHITE
from .eval import Evaluator
from .search import SearchEngine

class TDTrainer:
    def __init__(self, evaluator: Evaluator, alpha: float = 1e-4, lambda_val: float = 0.7, epsilon: float = 0.15,
                 search_time_ms: int = 100, search_depth: int = 3):
        self.evaluator = evaluator
        self.alpha = alpha  # Learning rate
        self.lambda_val = lambda_val  # TD(λ) parameter
        self.epsilon = epsilon  # Exploration rate
        self.search_engine = SearchEngine(evaluator)
        self.search_time_ms = search_time_ms
        self.search_depth = search_depth
    
    def train_episode(self) -> List[Tuple[Board, float, float]]:
        """Play one self-play episode and return transitions"""
        board = Board()
        transitions = []
        
        while not board.terminal():
            current_state = board.copy()
            current_features = self._get_features_dict(board, board.to_move)
            
            # Choose action (epsilon-greedy)
            if random.random() < self.epsilon:
                # Random move
                legal_moves = board.legal_moves()
                if legal_moves:
                    move = random.choice(legal_moves)
                else:
                    # Pass
                    board.to_move = -board.to_move
                    continue
            else:
                # Best move (shallow search)
                move, _, _ = self.search_engine.search_best_move(
                    board, board.to_move, max_time_ms=self.search_time_ms, max_depth=self.search_depth
                )
                if not move:
                    # Pass
                    board.to_move = -board.to_move
                    continue
            
            # Apply move
            board = board.apply(move)
            
            # Get reward and next state
            if board.terminal():
                winner = board.winner()
                if winner == board.to_move:
                    reward = 1.0
                elif winner == -board.to_move:
                    reward = -1.0
                else:
                    reward = 0.0
            else:
                reward = 0.0
            
            next_features = self._get_features_dict(board, board.to_move)
            
            transitions.append((current_state, current_features, reward, next_features))
        
        return transitions
    
    def update_weights(self, transitions: List[Tuple[Board, dict, float, dict]]):
        """Update weights using TD(λ)"""
        if not transitions:
            return
        
        # Initialize eligibility traces
        eligibility = {}
        for key in self.evaluator.weights:
            eligibility[key] = 0.0
        
        # Process transitions in reverse order
        for i in range(len(transitions) - 1, -1, -1):
            state, features, reward, next_features = transitions[i]
            
            # Current value
            current_value = self._evaluate_features(features)
            
            # Next value (0 if terminal)
            if i == len(transitions) - 1:
                next_value = 0.0
            else:
                next_value = self._evaluate_features(next_features)
            
            # TD error
            td_error = reward + next_value - current_value
            
            # Update eligibility traces
            for key in features:
                if key in self.evaluator.weights:
                    eligibility[key] = self.lambda_val * eligibility[key] + features[key]
            
            # Update weights
            import math
            for key in self.evaluator.weights:
                if key in eligibility:
                    new_val = self.evaluator.weights[key] + self.alpha * td_error * eligibility[key]
                    # Clamp and sanitize to avoid NaN/Inf blowups
                    if not math.isfinite(new_val):
                        new_val = 0.0
                    else:
                        new_val = max(min(new_val, 1e6), -1e6)
                    self.evaluator.weights[key] = new_val
    
    def _get_features_dict(self, board: Board, color: int) -> dict:
        """Get features as a dictionary"""
        features = self.evaluator._get_features(board, color)
        return features
    
    def _evaluate_features(self, features: dict) -> float:
        """Evaluate position using current weights"""
        score = 0.0
        
        # Calculate stage factor
        # For simplicity, assume mid-game
        mid = 0.5
        
        # Mid-game evaluation
        mid_score = (
            self.evaluator.weights["mobility"] * features.get("mobility", 0) +
            self.evaluator.weights["psqt"] * features.get("psqt", 0) +
            self.evaluator.weights["frontier"] * features.get("frontier", 0)
        )
        
        # End-game evaluation
        end_score = (
            self.evaluator.weights["parity"] * features.get("parity", 0) +
            self.evaluator.weights["corners"] * features.get("corners", 0) +
            self.evaluator.weights["corner_closeness"] * features.get("corner_closeness", 0)
        )
        
        return mid * mid_score + (1 - mid) * end_score
    
    def train(self, episodes: int = 1000, save_every: int = 100, out_path: str = "weights.json", log_every: int = 10):
        """Train for specified number of episodes"""
        import time
        start = time.time()
        print(f"Starting TD(λ) training for {episodes} episodes...")
        print(f"Parameters: α={self.alpha}, λ={self.lambda_val}, ε={self.epsilon}, search={self.search_depth}@{self.search_time_ms}ms")
        
        for episode in range(episodes):
            if episode % log_every == 0:
                elapsed = time.time() - start
                print(f"Episode {episode}/{episodes}  elapsed {elapsed:.1f}s", flush=True)
            
            # Play episode
            transitions = self.train_episode()
            
            # Update weights
            self.update_weights(transitions)
            
            # Periodic checkpoint
            if save_every and (episode + 1) % save_every == 0:
                self.evaluator.save_weights(out_path)
        
        # Save learned weights
        self.evaluator.save_weights(out_path)
        print(f"Training completed. Weights saved to {out_path}")
        
        # Print final weights
        print("\nFinal weights:")
        for key, value in self.evaluator.weights.items():
            print(f"  {key}: {value:.2f}")

def main():
    """Main training function with simple CLI arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Train Othello eval via TD(λ) self-play")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of self-play episodes")
    parser.add_argument("--alpha", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_val", type=float, default=0.7, help="TD(lambda)")
    parser.add_argument("--epsilon", type=float, default=0.15, help="Epsilon for exploration")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for self-play moves")
    parser.add_argument("--time_ms", type=int, default=100, help="Search time per move (ms)")
    parser.add_argument("--save_every", type=int, default=100, help="Save weights every N episodes")
    parser.add_argument("--out", type=str, default="weights.json", help="Output weights path")
    parser.add_argument("--log_every", type=int, default=10, help="Log progress every N episodes")
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = Evaluator()
    
    # Print initial weights
    print("Initial weights:")
    for key, value in evaluator.weights.items():
        print(f"  {key}: {value:.2f}")
    
    # Create trainer
    trainer = TDTrainer(
        evaluator,
        alpha=args.alpha,
        lambda_val=args.lambda_val,
        epsilon=args.epsilon,
        search_time_ms=args.time_ms,
        search_depth=args.depth,
    )
    
    # Train
    trainer.train(episodes=args.episodes, save_every=args.save_every, out_path=args.out, log_every=args.log_every)

if __name__ == "__main__":
    main()
