import time
import random
import csv
import statistics
import numpy as np
from src.Board import Board
from src.Colour import Colour

class HexSolver:
    def __init__(self):
        pass

    def find_winning_move_limited(self, board, player_colour, max_depth):
        best_score = -float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        # --- FIX 1: Correctly find empty tiles from Tile objects ---
        # We look at the .colour property, not the object itself
        empty_spots = [
            (r, c) 
            for r in range(board.size) 
            for c in range(board.size) 
            if board.tiles[r][c].colour is None
        ]
        
        # Shuffle for realistic testing
        random.shuffle(empty_spots)

        for idx in empty_spots:
            r, c = idx
            
            # --- FIX 2: Use None for empty ---
            board.tiles[r][c].colour = player_colour
            
            # Check win immediately (Ply 1)
            if board.has_ended(player_colour):
                score = 1
            else:
                # Only recurse if we have depth budget left
                if max_depth > 1:
                    score = self._minimax_recursive(board, False, player_colour, alpha, beta, max_depth - 1)
                else:
                    score = 0 

            # Reset to None
            board.tiles[r][c].colour = None
            
            if score > best_score:
                best_score = score
                best_move = (r, c)
            
            # Alpha-Beta Pruning
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        
        return best_move if best_score == 1 else None

    def _minimax_recursive(self, board, is_maximizing, root_player, alpha, beta, depth):
        current_player = root_player if is_maximizing else self._get_opponent(root_player)
        prev_player = self._get_opponent(current_player)

        # 1. Did the previous move win?
        if board.has_ended(prev_player):
            return -1 if is_maximizing else 1
        
        # 2. Depth Limit Reached?
        if depth == 0:
            return 0

        # --- FIX 1: Correct list comprehension for empty tiles ---
        empty_spots = [
            (r, c) 
            for r in range(board.size) 
            for c in range(board.size) 
            if board.tiles[r][c].colour is None
        ]
        
        if not empty_spots:
            return 0 

        if is_maximizing:
            max_eval = -float('inf')
            for idx in empty_spots:
                r, c = idx
                board.tiles[r][c].colour = current_player
                eval_score = self._minimax_recursive(board, False, root_player, alpha, beta, depth - 1)
                board.tiles[r][c].colour = None
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for idx in empty_spots:
                r, c = idx
                board.tiles[r][c].colour = current_player
                eval_score = self._minimax_recursive(board, True, root_player, alpha, beta, depth - 1)
                board.tiles[r][c].colour = None
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval

    def _get_opponent(self, colour):
        return Colour.BLUE if colour == Colour.RED else Colour.RED

# --- BENCHMARK UTILITIES ---

def fill_board_randomly(board, moves_to_play):
    board_size = board.size
    all_positions = [(r, c) for r in range(board_size) for c in range(board_size)]
    random.shuffle(all_positions)
    
    current_colour = Colour.RED
    for i in range(moves_to_play):
        if not all_positions: break
        r, c = all_positions.pop()
        board.tiles[r][c].colour = current_colour
        current_colour = Colour.BLUE if current_colour == Colour.RED else Colour.RED

def run_benchmark():
    BOARD_SIZE = 11
    TOTAL_TILES = BOARD_SIZE * BOARD_SIZE
    
    # --- CONFIGURATION ---
    # Depth 2 is safe to test on empty boards. 
    # Depth 3 is VERY slow on empty boards (avoid testing 120/3 unless you have time).
    test_depths = [4, 5, 10, 20] 
    
    # Start with Empty(120), then Mid(60), then End(20, 15, 10)
    test_empty_counts = [50, 40, 20, 15, 10]
    trials_per_case = 3
    
    solver = HexSolver()

    # Warmup
    print("Warming up...", end="", flush=True)
    dummy_board = Board(BOARD_SIZE)
    solver.find_winning_move_limited(dummy_board, Colour.RED, 1)
    print(" Done.")

    print(f"{'Empty':<6} | {'Depth':<6} | {'Avg Time':<12} | {'Max Time':<12}")
    print("-" * 65)
    
    results = []

    for empty_count in test_empty_counts:
        moves_played = TOTAL_TILES - empty_count
        
        for depth in test_depths:
            times = []
            
            for _ in range(trials_per_case):
                board = Board(BOARD_SIZE)
                fill_board_randomly(board, moves_played)
                
                start_time = time.perf_counter()
                solver.find_winning_move_limited(board, Colour.RED, depth)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            max_time = max(times)
            
            print(f"{empty_count:<6} | {depth:<6} | {avg_time:.6f}     | {max_time:.6f}")
            
            results.append({
                "Empty": empty_count,
                "Depth": depth,
                "Avg_Time": avg_time
            })

    with open("depth_benchmark_fixed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Empty", "Depth", "Avg_Time"])
        writer.writeheader()
        writer.writerows(results)
    
    print("\nResults saved to 'depth_benchmark_fixed.csv'")

if __name__ == "__main__":
    run_benchmark()