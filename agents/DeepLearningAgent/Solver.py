import numpy as np
import math
from src.Colour import Colour

class HexSolver:
    def __init__(self):
        self.COMPLEXITY_BUDGET = 110000000

    def find_winning_move(self, board, player_colour):
        """
        Dynamically adjusts search depth based on board emptiness.
        Returns a move (row, col) ONLY if a forced win is found within the depth limit.
        """
        # 1. Count Empty Tiles (Fastest way using your Tile objects)
        # Note: You can optimize this if your Board class tracks empty count, 
        # but this list comprehension is fast enough (~0.0001s).
        empty_indices = [
            (r, c) 
            for r in range(board.size) 
            for c in range(board.size) 
            if board.tiles[r][c].colour is None
        ]
        num_empty = len(empty_indices)

        # 2. Determine Search Depth based on your benchmarks
        depth = self._get_dynamic_depth(num_empty)

        # 3. ENDGAME: Full Solve
        # If board is very full, ignore depth limits and solve to the end.
        if num_empty <= 16:
            return self.solve_minimax(board, player_colour, max_depth=999, empty_indices=empty_indices)

        # 4. NORMAL: Limited Depth Search
        return self.solve_minimax(board, player_colour, max_depth=depth, empty_indices=empty_indices)

    def _get_dynamic_depth(self, num_empty):
        """
        Calculates depth to keep computational complexity roughly constant.
        Formula: Depth = log(Budget) / log(Branching_Factor)
        """
        if num_empty <= 1: 
            return 1
            
        # Logarithmic scaling formula
        raw_depth = math.log(self.COMPLEXITY_BUDGET) / math.log(num_empty)
        
        # We prefer to round down (int) to be safe, but you can round() if feeling aggressive.
        depth = int(raw_depth)
        
        # Hard limits
        # Even if the formula says Depth 1, we usually want at least Depth 2 to see immediate threats.
        return max(depth, 2)

    def solve_minimax(self, board, root_player, max_depth, empty_indices):
        # Initialize with losing score
        best_score = -float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        np.random.shuffle(empty_indices)

        for idx in empty_indices:
            r, c = idx
            
            board.tiles[r][c].colour = root_player
            
            # --- SCORING CHANGE: Prefer fast wins ---
            # If we win immediately, the score is very high (e.g., 1000)
            if board.has_ended(root_player):
                score = 1000
            else:
                if max_depth > 1:
                    # Pass '1' as current depth to the recursive function
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
            
            # Optimization: If found a win (score > 500 means guaranteed win), stop.
            if best_score > 500:
                return best_move
        
        # Only return a move if it's a guaranteed win (score > 500)
        return best_move if best_score > 500 else None

    def _minimax_recursive(self, board, is_maximizing, root_player, alpha, beta, depth_left, current_depth):
        current_player = root_player if is_maximizing else self._get_opponent(root_player)
        prev_player = self._get_opponent(current_player)

        # 1. Terminal State?
        if board.has_ended(prev_player):
            # If the PREVIOUS player just won:
            # - If maximizing (Root Player), it means Root Player won. Score = Positive.
            # - If minimizing (Opponent), it means Opponent won. Score = Negative.
            
            # Subtract current_depth so faster wins are worth MORE
            # Example: Win at depth 1 = 999. Win at depth 5 = 995.
            base_score = 1000 - current_depth
            
            return -base_score if is_maximizing else base_score
        
        if depth_left == 0:
            return 0

        # Optimization: Recalculate empty spots
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
                
                # Recurse: decrease depth_left, increase current_depth
                eval_score = self._minimax_recursive(
                    board, False, root_player, alpha, beta, depth_left - 1, current_depth + 1
                )
                
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
                
                eval_score = self._minimax_recursive(
                    board, True, root_player, alpha, beta, depth_left - 1, current_depth + 1
                )
                
                board.tiles[r][c].colour = None
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval

    def _get_opponent(self, colour):
        return Colour.BLUE if colour == Colour.RED else Colour.RED