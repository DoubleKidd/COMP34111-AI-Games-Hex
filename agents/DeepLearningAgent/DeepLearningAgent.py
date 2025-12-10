import torch
import os
import numpy as np
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from agents.DeepLearningAgent.model import AlphaZeroHexNet
from agents.DeepLearningAgent.mcts import MCTS
from agents.DeepLearningAgent.utils import encode_board

class DeepLearningAgent(AgentBase):
    """
    This is a deep learning agent based on alphazero architecture.
    It uses a trained neural network to evaluate board positions and select moves.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._board_size = 11
        
        # 1. Initialize the Network
        # Auto-detect device (CUDA, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.model = AlphaZeroHexNet(board_size=self._board_size).to(self.device)
        
        # 2. Try to load trained weights
        # We look for a file named "best_model.pth" in the same folder as this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "best_model.pth")
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            # print("AlphaZero: Loaded trained weights!")
        else:
            print("AlphaZero: WARNING - No weights found! Playing with random initialization.")
        
        self.model.eval()

        # 3. Initialize MCTS
        self.mcts = MCTS(self.model, cpu_ct=0.1)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """
        # --- HANDLE SWAP RULE (Turn 2) ---
        # If we are the second player (Turn 2), we check if the first move was strong, and steal it if so.
        if turn == 2:
            # Check the board value from the perspective of the player who just moved (Opponent)
            # We construct a board where we (current agent) are the "player to move"
            # But we want to know if the *previous* move was good for THEM.
            
            # Simple Heuristic for Swap:
            # If the network thinks the current state is bad for ME (Value < -0.2), 
            # it means it was good for the opponent. So we swap.
            
            # Encode board for ME
            input_tensor = encode_board(board, self.colour)
            input_tensor = torch.from_numpy(input_tensor).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                _, value = self.model(input_tensor)
            
            # Value is between -1 (I lose) and 1 (I win).
            # If value is low (e.g. < -0.1), it means the opponent's opening was strong.
            if value.item() < -0.1:
                return Move(-1, -1) # SWAP

        # --- STANDARD MOVE SELECTION ---
        
        # 1. Run MCTS Simulations
        # For a real game, you might want 100-200 simulations if time permits.
        # For now, 50 is fast and safe.
        simulations = 500
        
        
        # 2. Get the move probabilities (Temperature=0 for competitive play)
        # We copy inside MCTS, so just pass the board
        probs = self.mcts.get_action_prob(board, temp=0.01, simulations=simulations)
        best_action_index = np.argmax(probs)
        
        # 3. Convert Index to (Row, Col)
        
        row = best_action_index // self._board_size
        col = best_action_index % self._board_size

        return Move(row, col)