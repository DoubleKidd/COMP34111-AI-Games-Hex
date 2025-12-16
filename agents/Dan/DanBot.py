from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class DanBot(AgentBase):
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]
        self._score_board = [[100 for i in range(self._board_size)] for i in range(self._board_size)]
        self._board_store = None
    
    def analyse_board(self, board: Board) -> float:
        """Returns a float representing how close each player is to winning.
        -INF means red has won, 
        INF means blue has won
        """
        redscore = 0
        bluescore = 0

        redscore += analyse_paths(board, Colour.RED)
        bluescore += analyse_paths(board, Colour.BLUE)

        return bluescore - redscore

    def analyse_paths(self, board: Board, colour: Colour) -> list[dict]:
        """
        Analyse all connected paths of a given colour on the board.
        Returns a list of paths with their tiles, path length, and distance to walls.
        Each path is a dict: {'tiles': [(x,y), ...], 'length': int, 'dist_to_walls': (x, y)}
        """
        paths = []
        visited = [[False for _ in range(board.size)] for _ in range(board.size)]
        
        # Neighbour directions for hex grid
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1)]
        
        def dfs(x, y, current_path):
            visited[x][y] = True
            current_path.append((x, y))
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board.size and 0 <= ny < board.size:
                    if not visited[nx][ny] and board.tiles[nx][ny].colour == colour:
                        dfs(nx, ny, current_path)
        
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour == colour and not visited[i][j]:
                    path_tiles = []
                    dfs(i, j, path_tiles)
                    
                    # Compute distance to walls for the path endpoints
                    dist_to_walls_list = [self.distance_to_walls(x, y, colour) for x, y in path_tiles]
                    # For a simple heuristic, take min distance to start and min distance to end
                    dist_to_start = min(d[0] for d in dist_to_walls_list)
                    dist_to_end = min(d[1] for d in dist_to_walls_list)
                    
                    paths.append({
                        'tiles': path_tiles,
                        'length': len(path_tiles),
                        'dist_to_walls': (dist_to_start, dist_to_end)
                    })
        
        return paths

    
    def distance_to_walls(self, x: int, y: int, colour: Colour) -> tuple[int, int]:
        """
        Returns a tuple (distance_to_start, distance_to_end):
        - For RED: distance to top, distance to bottom
        - For BLUE: distance to left, distance to right
        """
        size = self._board_size
        if colour == Colour.RED:
            dist_to_start = x        # distance to top row (row 0)
            dist_to_end = size - 1 - x  # distance to bottom row
        elif colour == Colour.BLUE:
            dist_to_start = y        # distance to left column (col 0)
            dist_to_end = size - 1 - y  # distance to right column
        else:
            raise ValueError("Invalid colour for distance calculation")
        
        return dist_to_start, dist_to_end

        

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent move
        """

        if turn > 10:
            print(self.analyse_paths(board, self.colour))
            raise Exception

        # Remove opponents last move from choices
        if opp_move is not None and opp_move.x != -1:
            opponent_move = (opp_move.x, opp_move.y)
            self._choices.remove(opponent_move)

        if turn == 2:
            return Move(-1, -1)
        else:
            x, y = choice(self._choices)
            move = Move(x, y)

            # Remove move from choices and return move
            self._choices.remove((x, y))
            return move


    