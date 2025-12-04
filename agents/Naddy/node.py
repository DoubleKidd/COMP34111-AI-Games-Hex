from random import random
from typing import Self

from agents.Naddy.evaluate import action, evaluate
from src.Board import Board


class Node:
    def __init__(self, state: Board, parent: Self = None):
        """ Initializes a Node in the MCTS tree. """
        self.state = state
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
            child_states = self.action_func(self.state)

            for new_state in child_states:
                # Create a child for each state
                child_node = Node(new_state, parent=self)

                # Evaluate the state
                child_node.evaluate()
                self.children.append(child_node)

                # Early stopping for max reward
                if self.evaluate_func and child_node.reward >= 1:
                    return self.children  

        return self.children

    def best_child(self, policy_func: callable[[Self], Self]) -> Self | None:
        """ Selects the best child based on the provided policy function. """
        if not self.children:
            return None  

        # Apply policy function to select the best child
        return policy_func(self.children)  

    def backpropagate(self, reward: float):
        """Update reward and visits and propagate."""
        self.visits += 1

        self.reward += (reward - self.reward) / self.visits

        if self.parent:
            self.parent.backpropagate(reward)
