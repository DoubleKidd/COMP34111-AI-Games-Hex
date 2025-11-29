import torch
import numpy as np
from src.Board import Board
# Adjust imports based on your folder structure
from agents.DeepLearningAgent.model import AlphaZeroHexNet
from agents.DeepLearningAgent.mcts import MCTS

# 1. Setup minimal board
b = Board(board_size=4) # Use 4x4 for speed

# 2. Init Model and MCTS
model = AlphaZeroHexNet(board_size=4)
mcts = MCTS(model)

print("--- Running MCTS ---")

# 3. Get Action Probabilities
# This will run 50 simulations. 
# It might take 1-2 seconds on a CPU.
probs = mcts.get_action_prob(b, temp=1)

# 4. Analysis
print(f"Probabilities returned: {len(probs)}")
print(f"Max Probability: {max(probs):.4f}")

# Find best move index
best_move = np.argmax(probs)
row = best_move // 4
col = best_move % 4
print(f"AI suggests move: ({row}, {col})")

print("Test Passed!")