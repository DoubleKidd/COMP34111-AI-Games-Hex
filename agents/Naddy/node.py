from copy import deepcopy
import random
from typing import Callable, Self

from agents.Naddy.simulate import action, actions, simulate, state_to_result
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class Node:
    def __init__(
        self,
        state: Board,
        move: Move = None,
        colour: Colour = None,
        parent: Self | None = None
    ):
        """Initialise a node."""
        self.state = state
        self.move = move
        self.colour = colour
        self.parent = parent

        self.children: set[Self] = set()
        self.visits = 0
        self.reward = 0.5
        self.expanded = False

    def has_children(self) -> bool:
        return len(self.children) > 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0 and self.expanded

    def simulate(self) -> tuple[Board, float]:
        """Simulate from the node's state and return the outcome."""
        simulation = simulate(self.state, self.colour.opposite())
        result = state_to_result(simulation, self.colour)
        return simulation, result

    def expand(self) -> Self | None:
        """Generate child nodes from the current node's state."""
        self.expanded = True
        possible_moves = actions(self.state)
        if possible_moves == []:
            return None

        for new_move in possible_moves:
            new_state = deepcopy(self.state)
            new_state.set_tile_colour(new_move.x, new_move.y, self.colour.opposite())
            child_node = Node(new_state, move=new_move, colour=self.colour.opposite(), parent=self)
            self.children.add(child_node)

        return random.choice(list(self.children))

    def best_child(self, policy_func: Callable[[Self], Self]) -> Self | None:
        """Select the best child node based on the tree policy."""
        if not self.children:
            return None  

        return policy_func(self.children)

    def backpropagate(self, reward: float):
        """Update reward and visits and propagate."""
        self.visits += 1

        self.reward += (reward - self.reward) / self.visits

        if self.parent:
            self.parent.backpropagate(1 - reward)

    def visualise_tree(self, depth: int = 0):
        """Prints a visual representation of the tree from the given node."""
        indent = "_" * depth
        print(f"{indent}{self.move}, {self.colour}, R: {self.reward:.2f}, V: {self.visits}")
        for child in self.children:
            child.visualise_tree(depth + 1)
