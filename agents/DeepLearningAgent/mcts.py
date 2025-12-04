import math
import numpy as np
import torch
import copy
from src.Colour import Colour
from src.Board import Board
from src.Tile import Tile
# Adjust import if your utils file is named differently
from agents.DeepLearningAgent.utils import encode_board 

class MCTS:
    def __init__(self, model, cpu_ct=1.0):
        self.model = model
        self.cpu_ct = cpu_ct  # Controls exploration (higher = more exploring)
        
        # KEY DICTIONARIES
        self.Qsa = {}  # Quality (Win Rate) for State 's' taking Action 'a'
        self.Nsa = {}  # Number of times we took Action 'a' from State 's'
        self.Ns = {}   # Number of times we visited State 's'
        self.Ps = {}   # Policy (Probability) returned by Neural Net
        self.Es = {}   # Ended (Is the game over at state 's'?)
        self.Vs = {}   # Valid Moves mask for state 's'

    def get_action_prob(self, board, temp=1):
        """
        Runs simulations to determine the best move.
        Returns a probability distribution over all 121 moves.
        """
        # Run Simulations
        # For a student project, 25-50 simulations is usually enough for training.
        # For tournament play, bump this to 100+.
        for _ in range(50):
            # We must DEEP COPY the board because search() modifies it to simulate future turns
            # sim_board = copy.deepcopy(board) # Old slow way
            sim_board = self.clone_board(board) # New fast way
            self.search(sim_board)

        # 2. Count Visits
        s = self.get_board_string(board)
        counts = [self.Nsa.get((s, a), 0) for a in range(board.size * board.size)]

        # 3. Calculate Probabilities (Temperature)
        # temp=1 (Training): Softmax based on visits (keeps variety)
        # temp=0 (Tournament): Pick the absolute max visits (greedy)
        if temp == 0:
            best_a = np.argmax(counts)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board):
        """
        Performs one iteration: Selection -> Expansion -> Backprop
        """
        s = self.get_board_string(board)

        # --- STEP 1: CHECK IF GAME ENDED ---
        if s not in self.Es:
            self.Es[s] = board.has_ended(Colour.RED) or board.has_ended(Colour.BLUE)
        
        if self.Es[s]:
            # If the game is over, return the value.
            # Note: The search always evaluates from the perspective of the *current* player.
            # If the game ended, it means the *previous* player made a winning move.
            # Therefore, the current player has lost (-1).
            return -1

        # --- STEP 2: EXPANSION (Leaf Node) ---
        if s not in self.Ps:
            # 2a. Determine whose turn it is (for Encoding)
            # Count stones to find turn
            red_count = sum(1 for row in board.tiles for t in row if t.colour == Colour.RED)
            blue_count = sum(1 for row in board.tiles for t in row if t.colour == Colour.BLUE)
            current_colour = Colour.RED if red_count == blue_count else Colour.BLUE
            
            # 2b. Ask the Neural Network
            input_tensor = encode_board(board, current_colour)
            
            # Move tensor to the same device as the model
            device = next(self.model.parameters()).device
            input_tensor = torch.from_numpy(input_tensor).to(device).unsqueeze(0)
            # -------------------------------------------------------------
            
            self.model.eval()
            with torch.no_grad():
                policy, v = self.model(input_tensor)

            # 2c. Process Policy
            # Convert log_softmax back to probability
            policy = torch.exp(policy).data.cpu().numpy()[0]
            
            # NEW CODE ADDED HERE:
            if current_colour == Colour.BLUE:
                policy = self.transpose_policy(policy, board.size)

            # Mask invalid moves (spot is not empty)
            valids = self.get_valid_moves_mask(board)
            policy = policy * valids 
            
            # Re-normalize so sum is 1.0
            sum_policy = np.sum(policy)
            if sum_policy > 0:
                policy /= sum_policy 
            else:
                # Fallback if something went wrong (very rare)
                policy = policy + valids
                policy /= np.sum(policy)

            self.Ps[s] = policy
            self.Vs[s] = valids
            self.Ns[s] = 0
            
            # Return the Value estimate (-v because value is for the player who just moved)
            return -v.item()

        # --- STEP 3: SELECTION (PUCT Algorithm) ---
        valids = self.Vs[s]
        best_uct = -float('inf')
        best_act = -1

        # Check all valid moves to find the one with highest UCT score
        # UCT = Q(s,a) + U(s,a)
        for a in np.flatnonzero(valids):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpu_ct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpu_ct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

            if u > best_uct:
                best_uct = u
                best_act = a

        a = best_act
        
        # --- STEP 4: SIMULATE MOVE ---
        # Convert index 'a' (0..120) to (row, col)
        row = a // board.size
        col = a % board.size
        
        # Recalculate turn colour
        red_count = sum(1 for row in board.tiles for t in row if t.colour == Colour.RED)
        blue_count = sum(1 for row in board.tiles for t in row if t.colour == Colour.BLUE)
        current_colour = Colour.RED if red_count == blue_count else Colour.BLUE

        # TRANSFORM COORDINATES IF NEEDED
        # The Network thinks it's playing Red (Top-Bottom).
        # If we are Blue, the move (row, col) the network chose is actually (col, row) on the real board.
        # if current_colour == Colour.BLUE:
            # temp = row
            # row = col
            # col = temp

        # Apply the move to the simulation board
        board.set_tile_colour(row, col, current_colour)

        # RECURSE deeper
        v = self.search(board)

        # --- STEP 5: BACKPROPAGATION ---
        # Update Q and N
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
            
        self.Ns[s] += 1
        
        # Return negative value (because the opponent's gain is my loss)
        return -v

    def get_board_string(self, board):
        # Creates a unique string for the board state (for the dictionary keys)
        return str(board)

    def get_valid_moves_mask(self, board):
        # Returns array of 1s (valid) and 0s (occupied)
        mask = np.zeros(board.size * board.size)
        for r in range(board.size):
            for c in range(board.size):
                if board.tiles[r][c].colour is None:
                    mask[r * board.size + c] = 1
        return mask
    
    def transpose_policy(self, policy, size):
        # Takes the 121 probabilities, shapes them into a grid, flips the grid, and flattens it back.
        policy_2d = policy.reshape(size, size)
        policy_2d_T = policy_2d.T
        return policy_2d_T.flatten()

    def clone_board(self, original_board):
            """
            Fast manual clone of the board.
            Bypasses deepcopy by creating a fresh board and manually copying tile states.
            """
            # 1. Create a fresh board (this calls __init__, creating blank tiles)
            new_board = Board(original_board.size)
            
            # 2. Copy the winner status (if known) so we don't re-calculate it
            # We access the protected member _winner directly for speed
            new_board._winner = original_board._winner

            # 3. Copy the tile colours
            # We iterate through the grid and just copy the '.colour' attribute.
            # This is much faster than pickling/unpickling the whole Tile object.
            for r in range(original_board.size):
                for c in range(original_board.size):
                    # Only set if not None to save even more time
                    old_colour = original_board.tiles[r][c].colour
                    if old_colour is not None:
                        new_board.tiles[r][c].colour = old_colour
            
            return new_board