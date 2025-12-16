# --------------------------------------------------------------
# Basic Alpha beta 
# --------------------------------------------------------------
from random import choice
import time
import heapq
import random

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile


class BasicAB(AgentBase):
    _board_size = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [(i, j) for i in range(self._board_size)
                         for j in range(self._board_size)]

        self._time_limit = 175_000_000_000
        self._start_time = 0

        # ---- Zobrist ----
        random.seed(1337)
        self._zob = {(i, j, c): random.getrandbits(64)
                     for i in range(self._board_size)
                     for j in range(self._board_size)
                     for c in (0, 1, 2)}
        self._trans = {}   # TT

    # -----------------------------------------
    # Fast zobrist hash
    # -----------------------------------------
    def hash_board(self, board: Board):
        h = 0
        for i in range(board.size):
            for j in range(board.size):
                c = board.tiles[i][j].colour
                if c is None:
                    h ^= self._zob[(i, j, 0)]
                elif c == Colour.RED:
                    h ^= self._zob[(i, j, 1)]
                else:
                    h ^= self._zob[(i, j, 2)]
        return h

    # -----------------------------------------
    # FAST evaluation (no full Dijkstra)
    # -----------------------------------------
    def analyse_board(self, board: Board) -> float:
        # Simple connectivity heuristic:
        red_conn = 0
        blue_conn = 0

        for i in range(board.size):
            for j in range(board.size):
                c = board.tiles[i][j].colour
                if c == Colour.RED:
                    red_conn += 1
                elif c == Colour.BLUE:
                    blue_conn += 1

        return red_conn - blue_conn

    # -----------------------------------------
    # Alpha-beta (with time checks)
    # -----------------------------------------
    def alphabeta(self, board, depth, alpha, beta, maximizing):

        if time.perf_counter_ns() - self._start_time > self._time_limit:
            raise TimeoutError

        if depth == 0:
            return self.analyse_board(board)

        h = (self.hash_board(board), depth, maximizing)
        if h in self._trans:
            return self._trans[h]

        # Candidate move pruning
        moves = [(x, y) for (x, y) in self._choices
                 if board.tiles[x][y].colour is None]

        # Keep only top 12 closest to centre
        moves.sort(key=lambda m: abs(m[0]-5) + abs(m[1]-5))
        moves = moves[:12]

        best = float("-inf") if maximizing else float("inf")
        opp = Colour.RED if self.colour == Colour.BLUE else Colour.BLUE

        for x, y in moves:

            if time.perf_counter_ns() - self._start_time > self._time_limit:
                raise TimeoutError

            board.set_tile_colour(x, y, self.colour if maximizing else opp)

            val = self.alphabeta(board, depth-1,
                                 alpha, beta,
                                 not maximizing)

            board.set_tile_colour(x, y, None)

            if maximizing:
                best = max(best, val)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            else:
                best = min(best, val)
                beta = min(beta, best)
                if alpha >= beta:
                    break

        self._trans[h] = best
        return best

    # -----------------------------------------
    # Make move
    # -----------------------------------------
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:

        if opp_move and opp_move.x != -1:
            if (opp_move.x, opp_move.y) in self._choices:
                self._choices.remove((opp_move.x, opp_move.y))

        # PIE rule
        if turn == 2:
            if opp_move.x == 5 and opp_move.y == 5:
                return Move(-1, -1)

        self._start_time = time.perf_counter_ns()

        if turn < 10:
            MAXD = 2
        elif turn < 30:
            MAXD = 3
        else:
            MAXD = 4

        best_move = choice(self._choices)

        try:
            for depth in range(1, MAXD+1):

                if time.perf_counter_ns() - self._start_time > self._time_limit:
                    break

                best_val = float("-inf")
                best_here = best_move

                for (x, y) in self._choices:

                    if time.perf_counter_ns() - self._start_time > self._time_limit:
                        raise TimeoutError

                    board.set_tile_colour(x, y, self.colour)
                    val = self.alphabeta(board, depth,
                                         -1e18, 1e18,
                                         maximizing=False)
                    board.set_tile_colour(x, y, None)

                    if val > best_val:
                        best_val = val
                        best_here = (x, y)

                best_move = best_here

        except TimeoutError:
            pass

        self._choices.remove(best_move)
        return Move(best_move[0], best_move[1])
