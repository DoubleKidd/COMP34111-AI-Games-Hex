from random import random

from src.Board import Board
from src.Move import Move


def action(state: Board) -> list[Move]:
    """Generates possible moves from the given state."""
    return [
        Move(i, j) for i in range(state.size)
        for j
        in range(state.size)
        if state.tiles[i][j].colour is None
    ]


def evaluate(state: Board) -> float:
    """Evaluates the given state and returns a reward value."""
    # Placeholder: return rng
    return random()
