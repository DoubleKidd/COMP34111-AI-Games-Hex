from random import choice

from agents.Naddy.node import Node
from agents.Naddy.policy import ucb1_policy
from agents.Naddy.search import mcts
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

    _internal_board: Board

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._internal_board = None

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

        current_node = Node(
            state=board,
            move=opp_move,
            colour=self.colour.opposite(),
            turn=turn,
            parent=previous_node
        )
        best_move = mcts(
            current_node,
            ucb1_policy,
            time_per_move=3
        )
        # current_node.visualise_tree()

        previous_node = current_node
        return best_move
