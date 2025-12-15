from random import random
from typing import Self

from agents.Naddy.simulate import action, simulate
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class Node:
    def __init__(
        self,
        state: Board,
        move: Move = None,
        colour: Colour = None,
        parent: Self = None
    ):
        """Initialise a node."""
        self.state = state
        self.move = move
        self.colour = colour
        self.parent = parent

        self.children: set[Self] = set()
        self.visits = 0
        self.reward = 0
        self.expanded = False

        self.action_func = action
        self.simulate_func = simulate

    def has_children(self) -> bool:
        return len(self.children) > 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0 and self.expanded

    def simulate(self) -> float:
        """Simulate from the node's state and return the outcome."""
        if self.simulate_func:
            return self.simulate_func(self.state)
        return 0.0

    def expand(self) -> list[Self]:
        """Generate child nodes from the current node's state."""
        self.expanded = True
        new_move = self.action_func(self.state)

        new_state = self.state.copy()
        new_state.set_tile_colour(new_move.x, new_move.y, self.colour.opposite())
        child_node = Node(new_state, move=new_move, colour=self.colour.opposite(), parent=self)
        self.children.add(child_node)

        return child_node

    def best_child(self, policy_func: callable[[Self], Self]) -> Self | None:
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
