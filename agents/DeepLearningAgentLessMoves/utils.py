import numpy as np
from src.Colour import Colour

def encode_board(board, my_colour: Colour):
    """
    Converts the Board object into a (2, N, N) numpy tensor.
    
    Channels:
    0: My Stones
    1: Opponent Stones
    
    If my_colour is BLUE, we transpose the board so the AI always 
    thinks it is playing 'Vertical' (Red).
    """
    board_size = board.size
    # Shape: (2, 11, 11) - Float32 is standard for PyTorch
    encoded = np.zeros((2, board_size, board_size), dtype=np.float32)

    # We determine who the opponent is based on my_colour
    opp_colour = Colour.BLUE if my_colour == Colour.RED else Colour.RED

    for r in range(board_size):
        for c in range(board_size):
            tile = board.tiles[r][c]
            
            # If tile is empty, skip
            if tile.colour is None: # Assuming None is empty based on Board.py
                continue

            if my_colour == Colour.RED:
                # --- RED (Standard Perspective) ---
                # We want to connect Top-Bottom (Rows).
                # No transposition needed.
                if tile.colour == my_colour:
                    encoded[0, r, c] = 1 # My Stone
                elif tile.colour == opp_colour:
                    encoded[1, r, c] = 1 # Opp Stone

            else:
                # --- BLUE (Transposed Perspective) ---
                # We want to connect Left-Right (Cols).
                # We Swap (r, c) -> (c, r) so the AI sees it as a Vertical game.
                if tile.colour == my_colour:
                    encoded[0, c, r] = 1 # My Stone (Transposed)
                elif tile.colour == opp_colour:
                    encoded[1, c, r] = 1 # Opp Stone (Transposed)

    return encoded