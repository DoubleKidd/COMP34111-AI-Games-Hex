import subprocess
import sys
from src.Move import Move
from src.Colour import Colour
from src.Board import Board


class ExternalMCTS:
    def __init__(self, exe_path: str):
        self.proc = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def send(self, msg: str):
        if self.proc.poll() is not None:
            err = self.proc.stderr.read()
            print("C++ STDERR:", err)
            raise RuntimeError("C++ engine has exited")
        self.proc.stdin.write(msg)
        self.proc.stdin.flush()

    def recv(self) -> str:
        return self.proc.stdout.readline().strip()

    def close(self):
        self.proc.terminate()


def board_to_ascii(board: Board) -> str:
    rows = []
    for i in range(board.size):
        row = ""
        for j in range(board.size):
            tile = board.tiles[i][j].colour
            if tile is None:
                row += "."
            elif tile == Colour.RED:
                row += "R"
            else:
                row += "B"
        rows.append(row)
    return "\n".join(rows)


def query_mcts(
    engine: ExternalMCTS,
    board: Board,
    colour: Colour,
    turn: int,
    opp_move: Move | None
) -> Move:

    engine.send(f"SIZE {board.size}\n")
    engine.send(f"COLOUR {'RED' if colour == Colour.RED else 'BLUE'}\n")
    engine.send(f"TURN {turn}\n")

    if opp_move is None:
        engine.send("OPP NONE\n")
    else:
        engine.send(f"OPP {opp_move.x} {opp_move.y}\n")

    engine.send("BOARD\n")
    engine.send(board_to_ascii(board) + "\n")
    engine.send("END\n")

    reply = engine.recv()
    _, x, y = reply.split()
    return Move(int(x), int(y))
