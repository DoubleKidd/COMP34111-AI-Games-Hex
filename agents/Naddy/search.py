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
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
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
    time_per_move: float = 5.0,
    win_threshold: float = 1.0,
) -> Move:
    """Performs MCTS search and returns the best immediate next move for the AI."""
    logger.info(f"Performing MCTS with {time_per_move}s time limit as {root.colour.opposite()} on\n{root.state}")

    start_time = datetime.now()
    log_time = start_time
    iteration_count = 0
    time_left = time_per_move

    while time_left > 0:
        iteration_count += 1
        time_left = time_per_move - (datetime.now() - start_time).total_seconds()
        node = root
        logger.debug(f"Performing iteration {iteration_count} with {time_left:.03f}s left.")

        # Immediate win check
        # if node.reward >= win_threshold:
        #     logger.debug("Winning move found during initial selection.")
        #     return node.move

        # --- SELECTION ---
        while node is not None and node.has_children():
            child = selection_policy(node)
            if child is None:
                # logger.debug("Selection returned None.")
                break
            node = child

        if node is None:
            # logger.debug("Selection failed to find a node.")
            continue

        # logger.debug(f"Selected node move: {node.move}")
        # logger.debug(f"{node.get_state()}")

        log_time = take_time(log_time, "Selection")
        # --- EXPANSION ---
        if not node.expanded:
            child = node.expand()
            if child is None:
                # logger.debug("Expansion returned no children (terminal state?).")
                continue
            # logger.debug(f"Expanded node move: {child.move}")
            # logger.debug(f"{child.get_state()}")
        else:
            child = node
        log_time = take_time(log_time, "Expansion")

        # --- SIMULATION ---
        simulation, result, moves_played = child.simulate()
        # logger.debug(f"Simulated State:\n{simulation}\nResult: {result}")
        log_time = take_time(log_time, "Simulation")

        # --- BACKPROPAGATION ---
        child.backpropagate(result, moves_played)
        # logger.debug(f"Child reward {child.reward} ({child.result} / {child.visits})\nParent reward {child.parent.reward if child.parent else 'None'}")
        log_time = take_time(log_time, "Backpropagation")

    # After all iterations, select the best child of the root node
    logger.info(f"Completed {iteration_count} iterations in {(datetime.now() - start_time).total_seconds():.3f}s")

    if root.children:
        # Select best move: highest reward, then most visits as tiebreaker
        best_child = max(root.children, key=lambda c: (c.reward, c.visits))
        logger.info(f"Selecting move {best_child.move}.")
        logger.info(f"This move was rated {best_child.result} / {best_child.visits} ({best_child.reward}).")
        all_moves = '\n'.join(str(node) for node in sorted(root.children, key=lambda c: (c.reward, c.visits), reverse=True))
        logger.info(f"All moves:\n{all_moves}")
        return best_child.move
    else:
        # No moves available, return the root state
        return root.move
