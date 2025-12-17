import subprocess
import sys
import os
import copy
from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from agents.DeepLearningAgent.Solver import HexSolver # Import Solver

class KataHexAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.solver = HexSolver() # Initialize Solver
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(BASE_DIR, "hex3_27x_b28.bin.gz")
        self.config_path = os.path.join(BASE_DIR, "config.cfg")
        self.executable_path = os.path.join(BASE_DIR, "katahexexecutable")
        
        # Track if our internal KataHex board matches the game board
        self.board_synchronized = False 
        
        # --- LAUNCH KATAHEX ---
        self.cmd = [
            self.executable_path,
            "gtp",
            "-config", self.config_path,
            "-model", self.model_path
        ]
        self.process = None
        self._launch_katahex()

    def _launch_katahex(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=0
            )
            # New process = Board is empty, so we are NOT synchronized with the game yet
            self.board_synchronized = False 
            self._send_command_raw("boardsize 11")
            print(f"KataHex (re)launched for {self.colour}")
        except OSError as e:
            print(f"Error launching KataHex: {e}")
            sys.exit(1)

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_agent = cls.__new__(cls)
        memo[id(self)] = new_agent
        for k, v in self.__dict__.items():
            if k == 'process':
                setattr(new_agent, k, v) # Share the process
            else:
                setattr(new_agent, k, copy.deepcopy(v, memo))
        return new_agent

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:

        # first check for immediate winning/blocking moves using the solver
        winning_move = self.solver.find_winning_move(board, self.colour)
        if winning_move:
            print("Found immediate winning move via minmax solver.")
            self.immediate_move_found = True
            return Move(winning_move[0], winning_move[1])
        # Check for blocking opponent's winning move
        opponent_colour = Colour.BLUE if self.colour == Colour.RED else Colour.RED
        threat_move = self.solver.find_winning_move(board, opponent_colour)
        if threat_move:
            print(f"Blocking opponent's winning move at {threat_move}")
            self.immediate_move_found = True
            return Move(threat_move[0], threat_move[1])
        

        # --- STRATEGY SELECTION ---
        # We perform a FULL SYNC (Slow but Safe) only if:
        # 1. We just started/restarted the process (self.board_synchronized is False)
        # 2. It's very early in the game (Turn < 3) where SWAPS happen.
        # 3. The opponent just SWAPPED (opp_move is -1,-1).
        
        need_full_sync = (not self.board_synchronized) or (turn < 3)
        
        if opp_move is not None and opp_move.x == -1 and opp_move.y == -1:
            # Opponent swapped! Board state is messy, force a full sync.
            need_full_sync = True

        if need_full_sync:
            # --- SLOW PATH: FULL REPLAY ---
            self._sync_full_board(board)
            self.board_synchronized = True
        else:
            # --- FAST PATH: INCREMENTAL UPDATE ---
            if opp_move is not None:
                # 1. Determine Opponent Colour
                opp_colour = Colour.RED if self.colour == Colour.BLUE else Colour.BLUE
                # 2. Pass it explicitly to the helper
                self._play_single_stone(opp_move, opp_colour)

        # --- GENERATE MOVE ---
        my_gtp_color = self._colour_to_gtp(self.colour)
        response = self._send_command(f"genmove {my_gtp_color}")

        if not response:
            print("KataHex did not respond properly to genmove.")
            return self._random_valid_move(board)

        result = self._gtp_to_coord(response)
        
        if result == "swap-pieces":
            return Move(-1, -1)
        
        if result is None:
            print("KataHex did not respond properly to genmove.")
            return self._random_valid_move(board)

        row, col = result
        return Move(row, col)

    # --- NEW HELPER: Fast Single Update ---
    def _play_single_stone(self, move_obj: Move, colour: Colour):
        """Tells KataHex about just ONE move using the specific colour provided."""
        gtp_color = self._colour_to_gtp(colour)
        
        # HumanAgent uses Move(x, y) where x=row, y=col.
        # My _coord_to_gtp takes (row, col).
        gtp_coord = self._coord_to_gtp(move_obj.x, move_obj.y) 
        
        self._send_command(f"play {gtp_color} {gtp_coord}")

    # --- NEW HELPER: Slow Full Sync ---
    def _sync_full_board(self, board: Board):
        """Wipes KataHex and replays every stone on the board."""
        self._send_command("clear_board")
        for r in range(board.size):
            for c in range(board.size):
                tile = board.tiles[r][c]
                if tile.colour is not None:
                    gtp_color = self._colour_to_gtp(tile.colour)
                    gtp_coord = self._coord_to_gtp(r, c)
                    self._send_command(f"play {gtp_color} {gtp_coord}")

    # --- STANDARD HELPERS ---
    def _colour_to_gtp(self, colour_obj):
        if colour_obj == Colour.BLUE: return "white"
        return "black"

    def _coord_to_gtp(self, row, col):
        columns = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # Includes 'I'
        if col >= len(columns): return "PASS"
        return f"{columns[col]}{row + 1}"

    def _gtp_to_coord(self, gtp_string):
        gtp_string = gtp_string.strip().upper()
        if gtp_string == "SWAP": return "SWAP"
        if gtp_string in ["PASS", "RESIGN"]: return None 
        
        columns = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        try:
            col_char = gtp_string[0]
            row_val = int(gtp_string[1:])
            col = columns.index(col_char)
            row = row_val - 1
            return (row, col)
        except (ValueError, IndexError):
            return None

    def _send_command(self, command):
        """Safe wrapper that relaunches if dead."""
        if self.process is None or self.process.poll() is not None:
            print("KataHex died/missing. Relaunching...")
            self._launch_katahex()
            self.board_synchronized = False 
            
        return self._send_command_raw(command)

    def _send_command_raw(self, command):
        self.process.stdin.write(f"{command}\n")
        self.process.stdin.flush()
        
        response_lines = []
        while True:
            line = self.process.stdout.readline()
            if not line: break
            line = line.strip()
            if line == "": break
            response_lines.append(line)
            
        full_response = " ".join(response_lines)
        if full_response.startswith("="):
            return full_response[1:].strip()
        print(f"GTP Error: {full_response}")
        return None
    
    def _try_minimax_solver(self, board: Board) -> Move | None:
        """Attempts to use the internal HexSolver to find a forced win/loss."""
        winning_move = self.solver.find_winning_move(board, self.colour)
        if winning_move:
            print("Found immediate winning move via minmax solver.")
            return Move(winning_move[0], winning_move[1])
        print("No immediate winning move found, proceeding with MCTS.")

        # --- 2. DEFENSE (Saving Throw) ---
        # "If I do nothing, can the opponent force a win?"
        opponent_colour = Colour.BLUE if self.colour == Colour.RED else Colour.RED
        threat_move = self.solver.find_winning_move(board, opponent_colour)
        
        if threat_move:
            print(f"Solver: Blocking Opponent Threat at {threat_move}")
            # We steal the move they wanted to play
            print(f"Blocking opponent's winning move at {threat_move}")
            return Move(threat_move[0], threat_move[1])
        print("No immediate opponent threats detected, proceeding with MCTS.")
        return None
    
    def _random_valid_move(self, board: Board) -> Move:
        """Returns a random valid move on the board."""
        valid_moves = []
        for r in range(board.size):
            for c in range(board.size):
                if board.tiles[r][c].colour is None:
                    valid_moves.append(Move(r, c))
        if valid_moves:
            return choice(valid_moves)
        return Move(-1, -1)  # No valid moves available