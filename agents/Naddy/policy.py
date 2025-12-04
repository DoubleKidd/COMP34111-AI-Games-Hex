import random

from agents.Naddy.node import Node


def rand_policy(children: list[Node]):
    """Choose a random child node."""
    return random.choice(children)


def best_policy(children: list[Node]):
    """Choose the child node with the best value."""
    return max(children, key=lambda child: child.reward)
