from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class HumanAgent(AgentBase):
    """
    """

    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)

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
        while True:
            try:
                move_str = input(f"Your turn ({self.colour.name}). Enter your move as 'row,col': ")
                row, col = move_str.split(",")
                x, y = int(row), int(col)
                if turn == 2 and x == opp_move.x and y == opp_move.y:
                    return Move(-1, -1)
                elif not (0 <= x < self._board_size and 0 <= y < self._board_size):
                    print(f"Invalid move. Please enter values between 0 and {self._board_size - 1}.")
                elif board.tiles[x][y].colour is not None:
                    print("Invalid move. That tile is already occupied.")
                else:
                    return Move(x, y)
                
            except ValueError:
                print("Invalid input. Please enter your move as 'row,col' with valid integers.")
