from random import random
from typing import Self

from agents.Naddy.evaluate import action, evaluate
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class Node:
    def __init__(self, state: Board, move: Move = None, colour: Colour = None, parent: Self = None):
        """Initialise a node."""
        self.state = state
        self.move = move
        self.colour = colour
        self.parent = parent

        self.children: list[Self] = []
        self.visits = 0
        self.reward = 0
        self.expanded = False

        self.action_func = action
        self.evaluate_func = evaluate

    def has_children(self) -> bool:
        return len(self.children) > 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0 and self.expanded

    def evaluate(self) -> float:
        """Evaluate the node's state and return its reward."""
        if self.evaluate_func:
            self.reward = self.evaluate_func(self.state)
            return self.reward
        return 0.0

    def expand(self) -> list[Self]:
        """Generate child nodes from the current node's state."""
        self.expanded = True
        if self.action_func:
            possible_moves = self.action_func(self.state)

            for move in possible_moves:
                new_state = self.state.copy()
                new_state.set_tile_colour(move.x, move.y, self.colour.opposite())
                # Create a child for each state
                child_node = Node(new_state, move=move, colour=self.colour.opposite(), parent=self)

                # Evaluate the state
                child_node.evaluate()
                self.children.append(child_node)

                # Early stopping for max reward
                if self.evaluate_func and child_node.reward >= 1:
                    return self.children  

        return self.children

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
            self.parent.backpropagate(reward)
