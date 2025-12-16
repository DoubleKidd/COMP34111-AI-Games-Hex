from datetime import datetime
import logging
from pathlib import Path
from typing import Callable

from agents.Naddy.policy import *
from src.Move import Move

log_dir = Path(__file__).parent / "logs"
log_file = log_dir / f"mcts_{datetime.now().strftime('%d_%H%M%S')}.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s\n%(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)


def traverse_all(node: Node):
    """Traverse through all nodes below the given node."""
    nodes = [node]
    for child in node.children:
        nodes.extend(traverse_all(child))

    return nodes


def mcts(
    root: Node,
    selection_policy: Callable,
    iterations: int = 50,
    discount_factor: float = 0.9,
    win_threshold: float = 1.0,
) -> Move:
    """Performs MCTS search and returns the best immediate next move for the AI."""
    logger.info(f"Performing MCTS with {iterations} iterations as {root.colour.opposite()} on\n{root.state}.")

    for _ in range(iterations):
        node = root

        # Immediate win check
        # if node.reward >= win_threshold:
        #     logger.debug("Winning move found during initial selection.")
        #     return node.move

        # --- SELECTION ---
        while node is not None and node.has_children():
            child = selection_policy(node)
            if child is None:
                logger.debug("Selection returned None.")
                break
            node = child

        if node is None:
            logger.debug("Selection failed to find a node.")
            continue

        logger.debug(f"Selected node move: {node.move}")

        # --- EXPANSION ---
        if not node.has_children():
            child = node.expand()
            if child is None:
                logger.debug("Expansion returned no children (terminal state?).")
                continue
            logger.debug(f"Expanded node move: {child.move}")
            logger.debug(f"{child.state}")
        else:
            child = node

        # --- SIMULATION ---
        simulation, result = child.simulate()
        logger.debug(f"Simulated State:\n{simulation}\nResult: {result}")

        # --- BACKPROPAGATION ---
        child.backpropagate(result)
        logger.debug(f"Backpropagated value {result}\nChild reward {child.reward}\nParent reward {child.parent.reward if child.parent else 'None'}.")

    # After all iterations, select the best child of the root node
    if root.children:
        best_child = max(root.children, key=lambda c: c.reward)
        logger.info(f"Best Move: {best_child.move}\nReward: {best_child.reward}\nVisits: {best_child.visits}")
        return best_child.move
    else:
        # No moves available, return the root state
        return root.move
