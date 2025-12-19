from pathlib import Path

from src.AgentBase import AgentBase
from src.Move import Move
from src.Colour import Colour
from src.Board import Board

from .MCTSExternal import ExternalMCTS, query_mcts


EXECUTABLE = str(Path(__file__).parent / "MCTSBot")


class MCTSBot(AgentBase):
    def __init__(self, colour):
        super().__init__(colour)
        self.engine = ExternalMCTS(EXECUTABLE)

    def make_move(self, turn, board, opp_move):
        return query_mcts(
            engine=self.engine,
            board=board,
            colour=self.colour,
            turn=turn,
            opp_move=opp_move
        )

    def close(self):
        if hasattr(self, "engine") and self.engine:
            self.engine.close()
            self.engine = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["engine"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.engine is None:
            self.engine = ExternalMCTS(EXECUTABLE)


def __del__(self):
    if hasattr(self, "engine"):
        self.engine.close()
