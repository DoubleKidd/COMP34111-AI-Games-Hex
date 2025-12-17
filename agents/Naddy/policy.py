import math
import random

from agents.Naddy.node import Node


def rand_policy(node: Node):
    """Choose a random child node."""
    return random.choice(list(node.children)) if node.children != set() else None


def ucb1_policy(node: Node, exploration_constant: float = 1.414):
    """Choose child using UCB1 formula: reward + c * sqrt(ln(parent_visits) / child_visits)"""
    if not node.children:
        return None
    
    def ucb1_score(child: Node) -> float:
        if child.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        exploitation = child.reward
        exploration = exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
        return exploitation + exploration
    
    return max(node.children, key=ucb1_score)


def best_policy(node: Node):
    """Choose the child node with the best value."""
    return max(node.children, key=lambda child: child.reward)
