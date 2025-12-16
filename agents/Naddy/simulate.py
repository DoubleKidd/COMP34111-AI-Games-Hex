import random
from copy import deepcopy

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


def actions(state: Board) -> Move:
    """Generates possible moves from the given state."""
    return [
        Move(i, j) for i in range(state.size)
        for j
        in range(state.size)
        if state.tiles[i][j].colour is None
    ]


def action(state: Board) -> Move:
    """Generates possible moves from the given state."""
    return random.choice(actions(state))


def simulate(state: Board, colour: Colour, max_moves: int = 200) -> float:
    """Simulates random playout from the given state and returns reward."""
    sim_state = deepcopy(state)
    current_colour = colour
    moves_made = 0

    while moves_made < max_moves:
        move = action(sim_state)
        # If no moves remain, check who won
        if move is None:
            break

        sim_state.set_tile_colour(move.x, move.y, current_colour)

        current_colour = current_colour.opposite()
        moves_made += 1

    if sim_state.has_ended(colour):
        return 1.0
    elif sim_state.has_ended(colour.opposite()):
        return 0.0
    else:
        return 0.5
