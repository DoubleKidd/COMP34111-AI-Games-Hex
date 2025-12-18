import math
import random
from copy import deepcopy

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


def generate_actions(state: Board) -> Move:
    """Generates possible moves from the given state."""
    return [
        Move(i, j) for i in range(state.size)
        for j
        in range(state.size)
        if state.tiles[i][j].colour is None
    ]


def action_random(state: Board) -> Move | None:
    """Generates a random legal move."""
    moves = generate_actions(state)
    return random.choice(moves) if moves else None


def action_connect(state: Board, colour: Colour) -> Move:
    """Pick a move with Hex heuristics: center, adjacency, and 2-step links."""
    moves = generate_actions(state)
    if not moves:
        return None

    size = state.size
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    ring2 = set()
    for dx1, dy1 in dirs:
        ring2.add((dx1 * 2, dy1 * 2))
        for dx2, dy2 in dirs:
            ring2.add((dx1 + dx2, dy1 + dy2))

    def move_score(move: Move) -> float:
        center = (size - 1) / 2
        dist_penalty = abs(move.x - center) + abs(move.y - center)
        score = -dist_penalty / 4

        # Reward adjacency to own stones
        for dx, dy in dirs:
            nx, ny = move.x + dx, move.y + dy
            if 0 <= nx < size and 0 <= ny < size:
                neighbour_colour = state.tiles[nx][ny].colour
                if neighbour_colour == colour:
                    score += 3.0
                elif neighbour_colour == colour.opposite():
                    score += 1.0

        # Encourage 2-step links
        for dx, dy in ring2:
            nx, ny = move.x + dx, move.y + dy
            if 0 <= nx < size and 0 <= ny < size:
                neighbour_colour = state.tiles[nx][ny].colour
                if neighbour_colour == colour:
                    score += 1.5
                elif neighbour_colour == colour.opposite():
                    score += 0.5
        return score

    # Softmax-weighted sampling
    temperature = 1.5
    scores = [move_score(mv) for mv in moves]
    max_score = max(scores)
    weights = [math.exp((s - max_score) / temperature) for s in scores]

    return random.choices(moves, weights=weights, k=1)[0]


def state_to_result(state: Board, colour: Colour) -> float:
    """Checks who has won in the given state."""
    if state.has_ended(colour):
        return 1.0
    elif state.has_ended(colour.opposite()):
        return 0.0
    else:
        return 0.5


def simulate(state: Board, colour: Colour) -> tuple[Board, list[Move]]:
    """Simulates playout from the given state using a shuffled action list."""
    sim_state = deepcopy(state)
    current_colour = colour
    moves_played = []

    available_moves = generate_actions(sim_state)
    random.shuffle(available_moves)

    for move in available_moves:
        sim_state.set_tile_colour(move.x, move.y, current_colour)
        current_colour = current_colour.opposite()
        moves_played.append(move)

    return sim_state, moves_played
