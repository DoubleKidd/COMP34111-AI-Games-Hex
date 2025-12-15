import logging
from typing import Callable
from agents.Naddy.policy import *
from src.Move import Move

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def traverse_all(node: Node):
    """Traverse through all nodes below the given node."""
    nodes = [node]
    for child in node.children:
        nodes.extend(traverse_all(child))

    return nodes


def mcts_search(
    root: Node,
    iterations: int = 50,
    selection_policy: Callable | None = None,
    discount_factor: float = 0.9,
    win_threshold: float = 1.0,
) -> Move:
    """Performs MCTS search and returns the best immediate next move for the AI."""
    selection_policy = selection_policy or rand_policy

    for _ in range(iterations):
        # --- SELECTION ---
        node = root

        # Immediate win check
        if node.reward >= win_threshold:
            logger.debug("Winning state found during initial selection.")
            return node.state

        # Pick a leaf node
        while node.has_children():
            node = node.best_child(policy_func=selection_policy)

            if node is None:
                logger.debug("No valid child available during selection.")
                break

        # --- EXPANSION ---
        if node is not None and not node.has_children():
            child = node.expand()

            # Pick the newly expanded child node
            node = rand_policy(child)
            logger.debug(f"Expanded Node State:\n{node.state}\n")

        # --- SIMULATION & BACKPROPAGATION ---
        if node is not None:
            # simulate node state
            node.simulate()
            logger.debug(f"simulated State:\n{node.state}\nReward: {node.reward}")

            # Backpropagation
            node.backpropagate(node.reward)
            logger.debug(f"Backpropagated reward: {node.reward} to parent nodes.")

    # After all iterations, select the best child of the root node
    if root.children:
        best_child = max(root.children, key=lambda c: c.reward)
        final_move = best_child.move
        logger.debug(f"Best Move State after search:\n{final_move}")
        return final_move
    else:
        # No moves available, return the root state
        return root.move
