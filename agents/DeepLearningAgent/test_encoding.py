from src.Board import Board
from src.Colour import Colour
from utils import encode_board

# 1. Setup a board
b = Board(board_size=3) # Small 3x3 for easy reading

# 2. Simulate a game state
# Let's say RED plays at Top-Left (0,0)
b.tiles[0][0].colour = Colour.RED

# Let's say BLUE plays at Top-Right (0, 2)
b.tiles[0][2].colour = Colour.BLUE

print("--- RAW BOARD ---")
print(b.print_board())

# 3. Test RED Perspective (Should look normal)
encoded_red = encode_board(b, Colour.RED)
print("\n--- RED PERSPECTIVE (Me=Red) ---")
print("My Stones:\n", encoded_red[0])
print("Opp Stones:\n", encoded_red[1])

# 4. Test BLUE Perspective (Should be FLIPPED)
# Blue at (0,2) is "Top-Right".
# If we flip (0,2) -> (2,0), it becomes "Bottom-Left".
encoded_blue = encode_board(b, Colour.BLUE)
print("\n--- BLUE PERSPECTIVE (Me=Blue) ---")
print("My Stones (Should be at [2,0] due to flip):\n", encoded_blue[0])
print("Opp Stones (Should be at [0,0] - 0,0 flipped is still 0,0):\n", encoded_blue[1])