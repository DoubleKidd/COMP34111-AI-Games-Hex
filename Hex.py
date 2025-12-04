import argparse
import importlib
import sys
from random import choice

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hex",
        description="Run a game of Hex. By default, two naive agents will play.",
    )
    parser.add_argument(
        "-p1",
        "--player1",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help="Specify the player 1 agent, format: \"agents.GroupX.AgentFile AgentClassName\" .e.g. \"agents.Group0.NaiveAgent NaiveAgent\"",
    )
    parser.add_argument(
        "-p1Name",
        "--player1Name",
        default="Alice",
        type=str,
        help="Specify the player 1 name",
    )
    parser.add_argument(
        "-p2",
        "--player2",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help="Specify the player 2 agent, format: \"agents.GroupX.AgentFile AgentClassName\" .e.g. \"agents.Group0.NaiveAgent NaiveAgent\"",
    )
    parser.add_argument(
        "-p2Name",
        "--player2Name",
        default="Bob",
        type=str,
        help="Specify the player 2 name",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-b",
        "--board_size",
        type=int,
        default=11,
        help="Specify the board size",
    )
    parser.add_argument(
        "-l",
        "--log",
        nargs="?",
        type=str,
        default=sys.stderr,
        const="game.log",
        help=(
            "Save moves history to a log file,"
            "if the flag is present, the result will be saved to game.log."
            "If a filename is provided, the result will be saved to the provided file."
            "If the flag is not present, the result will be printed to the console, via stderr."
        ),
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=1,
        help="Specify the number of times to repeat the game between the two agents.",
    )
    args = parser.parse_args()
    p1_path, p1_class = args.player1.split(" ")
    p2_path, p2_class = args.player2.split(" ")
    p1 = importlib.import_module(p1_path)
    p2 = importlib.import_module(p2_path)

    p1_wins = 0
    p2_wins = 0

    player1_red = True

    for i in range(args.repeat):
        print(f"Starting game {i + 1}")

        if player1_red:
            g = Game(
                player1=Player(
                    name=args.player1Name,
                    agent=getattr(p1, p1_class)(Colour.RED),
                ),
                player2=Player(
                    name=args.player2Name,
                    agent=getattr(p2, p2_class)(Colour.BLUE),
                ),
                board_size=args.board_size,
                logDest=args.log,
                verbose=args.verbose,
            )
        else:
            g = Game(
                player1=Player(
                    name=args.player1Name,
                    agent=getattr(p1, p1_class)(Colour.RED),
                ),
                player2=Player(
                    name=args.player2Name,
                    agent=getattr(p2, p2_class)(Colour.BLUE),
                ),
                board_size=args.board_size,
                logDest=args.log,
                verbose=args.verbose,
            )

        player1_red = not player1_red

        g.run()
        
        winning_colour = g.board.get_winner()

        winner_name = None
        if g.player1.agent.colour == winning_colour:
            p1_wins += 1
            winner_name = args.player1Name
        elif g.player2.agent.colour == winning_colour:
            p2_wins += 1
            winner_name = args.player2Name

        print(f"Game {i + 1} finished. Winner: {winner_name if winner_name else 'N/A'}. Colour: {winning_colour if winning_colour else 'N/A'}")

    print("All games finished.")
    # print the fraction of wins for each player
    player1_win_rate = p1_wins / args.repeat
    player2_win_rate = p2_wins / args.repeat
    print(f"Player 1 ({args.player1Name}) win rate: {player1_win_rate}")
    print(f"Player 2 ({args.player2Name}) win rate: {player2_win_rate}")