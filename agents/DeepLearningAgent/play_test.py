from src.Board import Board
from src.Colour import Colour
from src.Game import Game # Assuming there is a Game class logic, or we simulate it manually
# Import your new agent
from agents.DeepLearningAgent.DeepLearningAgent import DeepLearningAgent
from agents.ValidNaiveAgent.NaiveAgent import ValidNaiveAgent # Or whatever opponent you have

def test_game():
    # 1. Initialize Agents
    red_agent = DeepLearningAgent(Colour.RED)
    blue_agent = ValidNaiveAgent(Colour.BLUE)
    
    board = Board(board_size=11)
    
    print("--- Starting Test Game ---")
    
    # Simulate a few turns
    # Turn 1: Red (AlphaZero)
    move1 = red_agent.make_move(1, board, None)
    print(f"Red (AlphaZero) plays: {move1.x}, {move1.y}")
    board.set_tile_colour(move1.x, move1.y, Colour.RED)
    
    # Turn 2: Blue (Naive)
    # (Simplified, skipping swap logic for this basic test)
    move2 = blue_agent.make_move(2, board, move1)
    print(f"Blue (Naive) plays: {move2.x}, {move2.y}")
    board.set_tile_colour(move2.x, move2.y, Colour.BLUE)
    
    # Turn 3: Red (AlphaZero)
    move3 = red_agent.make_move(3, board, move2)
    print(f"Red (AlphaZero) plays: {move3.x}, {move3.y}")
    board.set_tile_colour(move3.x, move3.y, Colour.RED)
    
    print("Agent is functioning correctly!")

if __name__ == "__main__":
    test_game()