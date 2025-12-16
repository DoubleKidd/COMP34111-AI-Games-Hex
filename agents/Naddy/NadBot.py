from random import choice

from agents.Naddy.node import Node
from agents.Naddy.search import mcts_search
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class NadBot(AgentBase):
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    _choices: list[Move]
    _board_size: int = 11
    _internal_board: Board

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]
        self._internal_board = None
    
    def analyse_board(self, board: Board) -> float:
        """Returns a float representing how close each player is to winning.
        -INF means red has won, 
        INF means blue has won
        """

        pass

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
            Move: The agent move
        """

        previous_node = None
        # Remove opponents last move from choices
        if opp_move is not None and opp_move.x != -1:
            opponent_move = (opp_move.x, opp_move.y)
            self._choices.remove(opponent_move)

        current_node = Node(
            state=board,
            parent=previous_node
        )
        best_move = mcts_search(
            current_node
        )

        # Remove move from choices and return move
        self._choices.remove((best_move.x, best_move.y))
        previous_node = current_node
        return best_move
