import torch
import torch.nn as nn
import torch.optim as optim
import random
from copy import deepcopy
from src.Colour import Colour
from src.Board import Board

# -----------------------------
# Bot with Heatmap Evaluation
# -----------------------------
class Bot:
    def __init__(self, board_size=11):
        self.board_size = board_size
        self.nn = nn.Sequential(
            nn.Linear(board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, board_size * board_size),
            nn.Tanh(),  # output in range [-1,1] representing tile value
        )
        # random initialization
        for p in self.nn.parameters():
            nn.init.uniform_(p, -1.0, 1.0)

    def clone(self):
        new_bot = Bot(self.board_size)
        new_bot.nn.load_state_dict(deepcopy(self.nn.state_dict()))
        return new_bot

    def mutate(self, rate=0.1):
        for p in self.nn.parameters():
            noise = torch.randn_like(p) * rate
            p.data += noise

# -----------------------------
# Heatmap-based Move
# -----------------------------
def bot_move(bot, board, colour):
    input_tensor = torch.zeros(board.size * board.size)
    for i in range(board.size):
        for j in range(board.size):
            idx = i * board.size + j
            tile = board.tiles[i][j]
            if tile.colour == colour:
                input_tensor[idx] = 1.0
            elif tile.colour == Colour.RED or tile.colour == Colour.BLUE:
                input_tensor[idx] = -1.0
            else:
                input_tensor[idx] = 0.0

    with torch.no_grad():
        output = bot.nn(input_tensor)
        output = output.view(board.size, board.size)

    # select highest scoring empty tile
    best_score = -float('inf')
    best_move = None
    for i in range(board.size):
        for j in range(board.size):
            if board.tiles[i][j].colour != Colour.RED and board.tiles[i][j].colour != Colour.BLUE:
                score = output[i, j].item()
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
    return best_move

# -----------------------------
# Fitness: Wins + Heatmap Evaluation
# -----------------------------
def evaluate_bot(bot, opponents=5, board_size=11):
    fitness = 0
    for _ in range(opponents):
        # random opponent (random moves)
        opp_bot = Bot(board_size)
        opp_bot.nn = nn.Sequential()  # empty network = random moves

        # random colour assignment
        if random.random() < 0.5:
            winner, board = play_game(bot, opp_bot, board_size)
            if winner == Colour.RED:
                fitness += 5  # weight win higher
            fitness += evaluate_heatmap(bot, board, Colour.RED)
        else:
            winner, board = play_game(opp_bot, bot, board_size)
            if winner == Colour.BLUE:
                fitness += 5
            fitness += evaluate_heatmap(bot, board, Colour.BLUE)
    return fitness

# Heatmap evaluation: virtual connection value
def evaluate_heatmap(bot, board, colour):
    score = 0
    for i in range(board.size):
        for j in range(board.size):
            tile = board.tiles[i][j]
            if tile.colour == colour:
                # closer to winning edge is better
                if colour == Colour.RED:
                    score += i / board.size
                else:
                    score += j / board.size
            elif tile.colour == Colour.RED or tile.colour == Colour.BLUE:
                # block opponent is good
                score -= 0.5
    return score

# -----------------------------
# Game play
# -----------------------------
def play_game(bot_red, bot_blue, board_size=11):
    board = Board(board_size)
    current_colour = Colour.RED
    while True:
        if current_colour == Colour.RED:
            move = bot_move(bot_red, board, Colour.RED)
        else:
            move = bot_move(bot_blue, board, Colour.BLUE)

        if move is None:
            return None, board

        x, y = move
        board.set_tile_colour(x, y, current_colour)

        if board.has_ended(current_colour):
            return current_colour, board

        current_colour = Colour.RED if current_colour == Colour.BLUE else Colour.BLUE

# -----------------------------
# Generational Training
# -----------------------------
def train_generations(generations=10, pop_size=100, elite=10, board_size=11):
    bots = [Bot(board_size) for _ in range(pop_size)]
    hall_of_fame = []

    for gen in range(generations):
        fitnesses = []
        for bot in bots:
            fit = evaluate_bot(bot, opponents=5, board_size=board_size)
            fitnesses.append(fit)

        sorted_bots = [b for _, b in sorted(zip(fitnesses, bots), key=lambda x: -x[0])]
        best_fitness = max(fitnesses)
        print(f"=== Generation {gen+1} ===")
        print(f"Best fitness: {best_fitness}")
        hall_of_fame.append(best_fitness)

        # next generation
        next_gen = []
        for i in range(elite):
            next_gen.append(sorted_bots[i].clone())
        while len(next_gen) < pop_size:
            parent = random.choice(sorted_bots[:elite]).clone()
            parent.mutate(rate=0.1)
            next_gen.append(parent)

        bots = next_gen

    # save best bot
    torch.save(sorted_bots[0].nn.state_dict(), "best_bot.pth")
    print("Training finished. Best bot saved as best_bot.pth")
    print("Hall of Fame fitness over generations:")
    for i, fit in enumerate(hall_of_fame):
        print(f"Gen {i+1}: fitness {fit}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train_generations(generations=10, pop_size=100, elite=10, board_size=11)
