import time
import random
import math
import statistics
import numpy as np

# --- IMPORT YOUR GAME CLASSES ---
# Adjust these imports if your folder structure is different
from src.Board import Board
from src.Colour import Colour

# --- THE SOLVER WITH LOGARITHMIC SCALING ---
class HexSolver:
    def __init__(self):
        # Target Complexity Budget: ~2 million nodes
        # This targets ~0.05s execution time
        self.COMPLEXITY_BUDGET = 210000000

    def find_winning_move(self, board, player_colour):
        # 1. Count Empty
        empty_indices = [
            (r, c) 
            for r in range(board.size) 
            for c in range(board.size) 
            if board.tiles[r][c].colour is None
        ]
        num_empty = len(empty_indices)

        # 2. Get Depth (For reporting purposes, we calculate it here too)
        depth = self._get_dynamic_depth(num_empty)

        # 3. ENDGAME: Full Solve
        if num_empty <= 12:
            return self.solve_minimax(board, player_colour, max_depth=999, empty_indices=empty_indices), depth, "Full Solve"

        # 4. NORMAL: Dynamic Depth
        move = self.solve_minimax(board, player_colour, max_depth=depth, empty_indices=empty_indices)
        return move, depth, "Dynamic"

    def _get_dynamic_depth(self, num_empty):
        if num_empty <= 1: return 1
        
        # LOGARITHMIC SCALING: Depth = log(Budget) / log(Branching_Factor)
        raw_depth = math.log(self.COMPLEXITY_BUDGET) / math.log(num_empty)
        depth = int(raw_depth)
        
        # Ensure at least depth 2 to see immediate threats
        return max(depth, 2)

    def solve_minimax(self, board, root_player, max_depth, empty_indices):
        best_score = -float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        # Shuffle for realistic testing
        random.shuffle(empty_indices)

        for idx in empty_indices:
            r, c = idx
            
            board.tiles[r][c].colour = root_player
            
            # Prefer fast wins scoring (1000 - depth)
            if board.has_ended(root_player):
                score = 1000
            else:
                if max_depth > 1:
                    score = self._minimax_recursive(board, False, root_player, alpha, beta, max_depth - 1, current_depth=1)
                else:
                    score = 0 

            board.tiles[r][c].colour = None
            
            if score > best_score:
                best_score = score
                best_move = (r, c)
            
            alpha = max(alpha, score)
            if beta <= alpha:
                break
                
            # If we found a guaranteed win, stop early
            if best_score > 500:
                return best_move
        
        return best_move if best_score > 500 else None

    def _minimax_recursive(self, board, is_maximizing, root_player, alpha, beta, depth_left, current_depth):
        current_player = root_player if is_maximizing else self._get_opponent(root_player)
        prev_player = self._get_opponent(current_player)

        if board.has_ended(prev_player):
            # Win Score = 1000 - distance
            base_score = 1000 - current_depth
            return -base_score if is_maximizing else base_score
        
        if depth_left == 0:
            return 0

        # Recalculate empty spots
        empty_spots = [
            (r, c) for r in range(board.size) for c in range(board.size) 
            if board.tiles[r][c].colour is None
        ]
        
        if not empty_spots:
            return 0 

        if is_maximizing:
            max_eval = -float('inf')
            for idx in empty_spots:
                r, c = idx
                board.tiles[r][c].colour = current_player
                eval_score = self._minimax_recursive(board, False, root_player, alpha, beta, depth_left - 1, current_depth + 1)
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
                eval_score = self._minimax_recursive(board, True, root_player, alpha, beta, depth_left - 1, current_depth + 1)
                board.tiles[r][c].colour = None
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval

    def _get_opponent(self, colour):
        return Colour.BLUE if colour == Colour.RED else Colour.RED

# --- BENCHMARK RUNNER ---

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

def run_scaling_test():
    BOARD_SIZE = 11
    TOTAL_TILES = BOARD_SIZE * BOARD_SIZE
    
    # We test a wide range of game stages to see the "Log" curve in action
    test_empty_counts = [120, 100, 80, 60, 50, 40, 30, 20, 15, 12]
    trials = 10
    
    solver = HexSolver()

    print(f"{'Empty':<6} | {'Algo Depth':<10} | {'Avg Time (s)':<12} | {'Max Time (s)':<12} | {'Notes'}")
    print("-" * 75)
    
    # Warmup
    dummy = Board(11)
    solver.find_winning_move(dummy, Colour.RED)

    for empty in test_empty_counts:
        moves_to_play = TOTAL_TILES - empty
        times = []
        depth_used = 0
        mode_used = ""
        
        for _ in range(trials):
            board = Board(BOARD_SIZE)
            fill_board_randomly(board, moves_to_play)
            
            start = time.perf_counter()
            # The modified find_winning_move returns extra info for debugging
            _, d, m = solver.find_winning_move(board, Colour.RED)
            end = time.perf_counter()
            
            times.append(end - start)
            depth_used = d
            mode_used = m
            
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        # Formatting
        depth_str = f"{depth_used}"
        if mode_used == "Full Solve":
            depth_str = "ALL"
        
        print(f"{empty:<6} | {depth_str:<10} | {avg_time:.6f}     | {max_time:.6f}     | {mode_used}")

if __name__ == "__main__":
    run_scaling_test()