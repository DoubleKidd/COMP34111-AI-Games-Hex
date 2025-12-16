from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

RED = Colour.RED
BLUE = Colour.BLUE

red_dydx = [
    (-2, 1),                # U
    (-1, -1),               # UL
    (-1, 2),                # UR
    (1, -2),                # DL
    (1, 1),                 # DR
    (2, 1),                 # D
]

red_dydx_empties = [
    ((-1, 0), (-1, 1)),     # U
    ((0, -1), (-1, 0)),     # UL
    ((-1, 1), (0, 1)),      # UR
    ((1, -1), (0, -1)),     # DL
    ((0, 1), (1, 0)),       # DR
    ((1, 0), (1, -1)),      # D
]

blue_dydx = [
    (-1, 2),                # UR
    (1, 1),                 # DR
    (-2, 1),                # U
    (2, -1),                # D
    (-1, -1),               # UL
    (1, -2),                # DL
]

blue_dydx_empties = [
    ((-1, 1), (0, 1)),      # UR
    ((0, 1), (1, 0)),       # DR
    ((-1, 0), (-1, 1)),     # U
    ((1, 0), (1, -1)),      # D
    ((0, -1), (-1, 0)),     # UL
    ((1, -1), (0, -1)),     # DL
]

class BridgeBot(AgentBase):
    """
    Bridger
    """

    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]
        self.virtuals = []
        self.downward = False
        self.extended = False
        self.initial_delay = 0
        self.anchor = None
    
    def in_board(self, x, y):
        return -1 < x < self._board_size and -1 < y < self._board_size
    
    def calculate_next_move(self, board: Board) -> Move:
        """
        By this time, we either have a connection downwards to the wall and a tile 
        at (7, 2) for RED or (8, 3) for BLUE or a tile at (1, 9) and are going downward.
        """
        x, y = self.anchor
        x0 = y0 = x1 = y1 = x2 = y2 = 0
        tiles = board.tiles

        for i in range(6):
            if self.colour == RED:
                x0, y0 = red_dydx[i]            # The big one
                empty1, empty2 = red_dydx_empties[i]
                x1, y1 = empty1                 # Little one
                x2, y2 = empty2                 # Little two
            else:
                x0, y0 = blue_dydx[i]            # The big one
                empty1, empty2 = blue_dydx_empties[i]
                x1, y1 = empty1                 # Little one
                x2, y2 = empty2                 # Little two          
            
            # Invert if going downward:
            if self.downward:
                x0 = -x0
                y0 = -y0
                x1 = -x1
                y1 = -y1
                x2 = -x2
                y2 = -y2
            
            nx = x + x0
            ny = y + y0

            nx1 = x + x1
            ny1 = y + y1

            nx2 = x + x2
            ny2 = y + y2

            try:
                if (
                    not self.in_board(nx, ny) or
                    not self.in_board(nx1, ny1) or
                    not self.in_board(nx2, ny2)
                ):
                    # Bad juju
                    # Check if virtual win!!
                        if self.colour == RED and (ny < 0 or ny >= self._board_size):
                            if tiles[nx][ny].colour == None:
                                return self.close_connections()
                            else:
                                return self.returnRandomMove(board)
                        elif self.colour == BLUE and (nx < 0 or nx >= self._board_size):
                            if tiles[nx][ny].colour == None:
                                return self.close_connections()
                            else:
                                return self.returnRandomMove(board)
                        
                        #Uhoh
                        else:
                            # THIS IS ALMOST CERTAINLY WRONG, BUT IT WORKS IN 95% of CASES
                            if tiles[nx][ny].colour == None:
                                return self.close_connections()
                            else:
                                return self.returnRandomMove(board)
                elif (
                    not self.in_board(nx, ny) and
                    self.in_board(nx1, ny1)
                ):
                    if tiles[nx1][ny1].colour == None and tiles[nx2][ny2].colour == None:
                        self.virtuals.append(((nx1, ny1), (nx2, ny2)))
                        return self.close_connections()
            except:
                return self.returnRandomMove(board)
                

            
            # Check if they are all empty
            if tiles[nx][ny].colour == None and tiles[nx1][ny1].colour == None and tiles[nx2][ny2].colour == None:
                move = Move(nx, ny)
                self.virtuals.append(((nx1, ny1), (nx2, ny2)))
                self.anchor = (nx, ny)
                try:
                    if tiles[nx][ny].colour == None:
                        return move
                    else:
                        move = self.close_connections()
                        # check if tile is free
                        if tiles[move.x][move.y].colour == None:
                            return move
                        else:
                            return self.returnRandomMove(board)
                except:
                    return self.returnRandomMove(board)
    
    def returnRandomMove(self, board) -> Move:
        #  list of available moves in board without using self._choices
        available_moves = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
            if board.tiles[i][j].colour is None
        ]
        x, y = choice(available_moves)
        return Move(x, y)

    def close_connections(self):
        move = None
        tmplist = []
        for i, (empty1, empty2) in enumerate(self.virtuals):
            if i == 0:
                x, y = empty1
                move = Move(x, y)
            else:
                tmplist.append((empty1, empty2))
        
        self.virtuals = tmplist.copy()
        return move

    def check_virtuals(self, board: Board) -> Move:
        tiles = board.tiles
        tmplist = []
        move = None
        for empty1, empty2 in self.virtuals:
            x1, y1 = empty1
            x2, y2 = empty2 

            if tiles[x1][y1].colour != None:
                self.initial_delay += 1
                move = Move(x2, y2)

            elif tiles[x2][y2].colour != None:
                self.initial_delay += 1
                move = Move(x1, y1)
            
            elif tiles[x1][y1].colour == None and tiles[x2][y2].colour == None:
                tmplist.append(((x1, y1), (x2, y2)))

        self.virtuals = tmplist.copy()

        return move

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

        # Claim any broken virtuals
        move = self.check_virtuals(board)
        if move is not None:
            return move

        # First turn:
        if turn == 1:
            if self.colour == RED:
                return Move(7, 2)
            else:
                return Move(8, 3)

        # Second turn
        elif turn == 2:
            # Take if these tiles if opponent took them
            if self.colour == RED and opp_move == Move(7, 2) or self.colour == BLUE and opp_move == Move(8, 3):
                return Move(-1, -1)
            
            elif self.colour == RED:
                return Move(7, 2)
            
            elif self.colour == BLUE:
                return Move(8, 3)

        # Third turn
        elif turn == 3 or turn == 4:
            tiles = board.tiles

            if opp_move == Move(-1, -1):
                self.downward == True

                if self.colour == RED:
                    self.virtuals.append(((0, 9), (0, 10)))
                else:
                    self.virtuals.append(((0, 10), (1, 10)))

                return Move(1, 9)

            elif tiles[9][1].colour == None:

                if self.colour == RED:
                    self.virtuals.append(((10, 0), (10, 1)))
                    self.virtuals.append(((8, 2), (8, 1)))
                else:
                    self.virtuals.append(((10, 0), (9, 0)))
                    self.virtuals.append(((8, 2), (9, 2)))

                return Move(9, 1)
            
            elif self.colour == RED and tiles[7][2].colour == None:
                self.virtuals.append(((8, 2), (8, 1)))
                self.extended = True
                return Move(7, 2)
            
            elif self.colour == BLUE and tiles[8][3].colour == None:
                self.virtuals.append(((8, 2), (9, 2)))
                self.extended = True
                return Move(8, 3)

            else:
                self.downward = True

                if self.colour == RED:
                    self.virtuals.append(((0, 9), (0, 10)))
                else:
                    self.virtuals.append(((0, 10), (1, 10)))

                return Move(1, 9)
        
        # Haven't got a successful connection yet
        elif self.extended and not self.downward and (turn == 5 + self.initial_delay or turn == 6 + self.initial_delay):
            tiles = board.tiles

            if self.colour == RED and tiles[9][4] == None:
                self.virtuals.append(((9, 3), (8, 4)))
                return Move(9, 4)

            elif self.colour == BLUE and tiles[6][1] == None:
                self.virtuals.append(((7, 1), (6, 2)))
                return Move(6, 1)
            
            elif self.colour == RED:
                self.virtuals.append(((9, 2), (9, 3)))
                return Move(10, 2)
            
            elif self.colour == BLUE:
                self.virtuals.append(((8, 1), (7, 1)))
                return Move(8, 0)

        # Other turns
        else:
            if self.anchor == None:
                if self.downward:
                    self.anchor = (1, 9)
                elif self.colour == RED:
                    self.anchor = (7, 2)
                else:
                    self.anchor = (8, 3)
                
            move = self.calculate_next_move(board)
            return move


    
