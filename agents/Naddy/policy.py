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
            return float('inf')
        exploitation = child.reward
        exploration = exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
        return exploitation + exploration

    return max(node.children, key=ucb1_score)


def rave_ucb_policy(node: Node, exploration_constant: float = 1.414, rave_constant: float = 300):
    """Choose child using RAVE-UCB: combines UCB1 with RAVE (Rapid Action Value Estimation)."""
    if not node.children:
        return None

    def rave_ucb_score(child: Node) -> float:
        if child.visits == 0:
            return float('inf')

        # Standard UCB1 value
        exploitation = child.reward
        exploration = exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
        ucb_value = exploitation + exploration

        # RAVE value
        move_key = str(child.move)
        if move_key in node.rave_visits and node.rave_visits[move_key] > 0:
            rave_value = node.rave_reward[move_key]
            
            beta = node.rave_visits[move_key] / (
                node.rave_visits[move_key] + child.visits + 
                4 * rave_constant * node.rave_visits[move_key] * child.visits
            )

            # Combine UCB and RAVE
            combined_value = (1 - beta) * ucb_value + beta * rave_value
        else:
            combined_value = ucb_value

        return combined_value

    return max(node.children, key=rave_ucb_score)


def best_policy(node: Node):
    """Choose the child node with the best value."""
    return max(node.children, key=lambda child: child.reward)
