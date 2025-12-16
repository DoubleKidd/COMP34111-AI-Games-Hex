from datetime import datetime
import logging
from pathlib import Path
from typing import Callable

from agents.Naddy.policy import *
from src.Move import Move

log_dir = Path(__file__).parent / "logs"
log_file = log_dir / f"mcts_{datetime.now().strftime('%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
        node = root

        # Immediate win check
        if node.reward >= win_threshold:
            logger.debug("Winning move found during initial selection.")
            return node.move

        # --- SELECTION ---
        while node is not None and node.has_children():
            child = node.best_child(policy_func=selection_policy)
            if child is None:
                logger.debug("Selection returned None.")
                break
            node = child

        if node is None:
            logger.debug("Selection failed to find a node.")
            continue

        # --- EXPANSION ---
        if not node.has_children():
            child = node.expand()
            if child is None:
                logger.debug("Expansion returned no children (terminal state?).")
                continue
            logger.debug(f"Expanded Node State:\n{node.state}\n")
        else:
            child = node

        # --- SIMULATION ---
        result = child.simulate()
        logger.debug(f"Simulated State:\n{child.state}\nResult: {result}")

        # --- BACKPROPAGATION ---
        child.backpropagate(result)
        logger.debug(f"Backpropagated value: {result} to parent nodes.")

    # After all iterations, select the best child of the root node
    if root.children:
        best_child = max(root.children, key=lambda c: c.reward)
        final_move = best_child.move
        logger.debug(f"Best Move State after search:\n{final_move}")
        return final_move
    else:
        # No moves available, return the root state
        return root.move
