from datetime import datetime
import logging
from pathlib import Path
from typing import Callable

from agents.Naddy.policy import *
from src.Move import Move

log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
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


def take_time(last_time: datetime, msg: str) -> datetime:
    curr_time = datetime.now()
    logger.debug(f"{msg} took {(curr_time - last_time).microseconds / 1000}ms")
    return curr_time


def mcts(
    root: Node,
    selection_policy: Callable,
    iterations: int = 50,
    discount_factor: float = 0.9,
    win_threshold: float = 1.0,
) -> Move:
    """Performs MCTS search and returns the best immediate next move for the AI."""
    logger.info(f"Performing MCTS with {iterations} iterations as {root.colour.opposite()} on\n{root.state}")

    last_time = datetime.now()
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
        logger.debug(f"\n{node.get_state()}")

        last_time = take_time(last_time, "Selection")
        # --- EXPANSION ---
        if not node.expanded:
            child = node.expand()
            if child is None:
                logger.debug("Expansion returned no children (terminal state?).")
                continue
            logger.debug(f"Expanded node move: {child.move}")
            logger.debug(f"\n{child.get_state()}")
        else:
            child = node
        last_time = take_time(last_time, "Expansion")

        # --- SIMULATION ---
        simulation, result = child.simulate()
        logger.debug(f"Simulated State:\n{simulation}\nResult: {result}")
        last_time = take_time(last_time, "Simulation")

        # --- BACKPROPAGATION ---
        child.backpropagate(result)
        logger.debug(f"Child reward {child.reward} ({child.reward} / {child.visits})\nParent reward {child.parent.reward if child.parent else 'None'}")
        last_time = take_time(last_time, "Backpropagation")

    # After all iterations, select the best child of the root node
    if root.children:
        best_child = max(root.children, key=lambda c: c.reward)
        logger.info(f"Best Move: {best_child.move}\nReward: {best_child.reward}\nVisits: {best_child.visits}")
        return best_child.move
    else:
        # No moves available, return the root state
        return root.move
