import random

from agents.Naddy.node import Node


def rand_policy(node: Node):
    """Choose a random child node."""
    return random.choice(list(node.children)) if node.children != set() else None


def best_policy(node: Node):
    """Choose the child node with the best value."""
    return max(node.children, key=lambda child: child.reward)
