from copy import deepcopy
import random
from typing import Callable, Self

from agents.Naddy.simulate import action_random, action_connect, generate_actions, simulate, state_to_result
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class Node:
    def __init__(
        self,
        state: Board,
        move: Move = None,
        colour: Colour = None,
        turn: int = 0,
        parent: Self | None = None
    ):
        """Initialise a node."""
        self.state = state
        self.move = move
        self.colour = colour
        self.turn = turn
        self.parent = parent

        self.children: set[Self] = set()
        self.expanded = False
        self.visits = 0
        self.result = 0
        self.rave_result = {}
        self.rave_visits = {}

    def has_children(self) -> bool:
        return len(self.children) > 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0 and self.expanded

    def simulate(self) -> tuple[Board, float]:
        """Simulate from the node's state and return the outcome."""
        simulation, moves_played = simulate(self.get_state(), self.colour.opposite())
        result = state_to_result(simulation, self.colour)
        return simulation, result, moves_played

    def expand(self) -> Self | None:
        """Generate child nodes from the current node's state."""
        self.expanded = True
        possible_moves = generate_actions(self.get_state())
        if possible_moves == []:
            return None

        for new_move in possible_moves:
            child_node = Node(
                state=None,
                move=new_move,
                colour=self.colour.opposite(),
                turn=self.turn+1,
                parent=self
            )
            self.children.add(child_node)

        chosen_child = random.choice(list(self.children))
        return chosen_child

    def get_state(self) -> Board:
        """Reconstruct state from parent's state and move."""
        if self.state is not None:
            return self.state
        if self.parent:
            self.state = deepcopy(self.parent.get_state())
            self.state.set_tile_colour(self.move.x, self.move.y, self.colour)
        return self.state

    def best_child(self, policy_func: Callable[[Self], Self]) -> Self | None:
        """Select the best child node based on the tree policy."""
        if not self.children:
            return None  

        return policy_func(self.children)

    def backpropagate(self, result: float, moves_played: list[Move] = None):
        """Update reward and visits and propagate, including RAVE statistics."""
        self.visits += 1
        self.result += result

        # RAVE statistics
        if moves_played:
            for move in moves_played:
                move_key = str(move)
                if move_key not in self.rave_visits:
                    self.rave_visits[move_key] = 0
                    self.rave_result[move_key] = 0
                self.rave_visits[move_key] += 1
                self.rave_result[move_key] += result

        if self.parent:
            self.parent.backpropagate(1 - result, moves_played)

    def visualise_tree(self, depth: int = 0):
        """Prints a visual representation of the tree from the given node."""
        indent = "_" * depth
        print(f"{indent}{self.move}, {self.colour}, R: {self.reward:.2f}, V: {self.visits}")
        for child in self.children:
            child.visualise_tree(depth + 1)

    def __repr__(self):
        return f"Node :: {self.move} with {self.colour} on T{self.turn} :: {self.result} / {self.visits} ({self.reward:.2f})"

    @property
    def reward(self): return self.result / self.visits if self.visits > 0 else 0.0

    @property
    def rave_reward(self): return {move_key: self.rave_result[move_key] / self.rave_visits[move_key] for move_key in self.rave_visits}
