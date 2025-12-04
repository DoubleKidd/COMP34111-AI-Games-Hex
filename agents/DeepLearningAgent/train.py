import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
import copy

from src.Board import Board
from src.Colour import Colour
from agents.DeepLearningAgent.model import AlphaZeroHexNet
from agents.DeepLearningAgent.mcts import MCTS
from agents.DeepLearningAgent.utils import encode_board

# --- STANDALONE WORKER FUNCTION ---
def self_play_worker(args):
    """
    Independent worker that plays a single game.
    Args:
        model_path: STRING path to the weights file (Safe!)
        board_size: Size of the board
        device_name: 'cuda' or 'cpu'
        simulations: Number of MCTS simulations per move
    """
    model_path, board_size, device_name, simulations = args
    
    # We wrap the whole process in a try/except block to catch errors safely
    try:
        # 1. Re-initialize model locally
        device = torch.device(device_name)
        local_model = AlphaZeroHexNet(board_size=board_size).to(device)
        
        # LOAD FROM FILE instead of memory (Bypasses CUDA shared memory crashes)
        if os.path.exists(model_path):
            local_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
        local_model.eval()
        
        # 2. Setup MCTS
        mcts = MCTS(local_model, cpu_ct=1.0)
        
        # 3. Play Game
        board = Board(board_size=board_size)
        train_examples = []
        current_colour = Colour.RED
        
        while True:
            # Get action probabilities
            # We hardcode temp=1 for training exploration
            probs = mcts.get_action_prob(board, temp=1)
            
            canonical_board = encode_board(board, current_colour)
            train_examples.append([canonical_board, probs, current_colour])
            
            action_index = np.random.choice(len(probs), p=probs)
            row = action_index // board_size
            col = action_index % board_size
            
            board.set_tile_colour(row, col, current_colour)
            
            if board.has_ended(current_colour):
                winner = current_colour
                break
                
            current_colour = Colour.BLUE if current_colour == Colour.RED else Colour.RED

        # 4. Format Data
        final_data = []
        for x in train_examples:
            board_state, policy, player = x
            value = 1 if player == winner else -1
            final_data.append((board_state, policy, value))
            
        return final_data

    except Exception as e:
        # This catches the error so the main process doesn't hang
        print(f"Worker Error: {e}")
        return []

class AlphaZeroTrainer:
    def __init__(self, board_size=6):
        self.board_size = board_size
        
        # Detect Mac M2 (MPS) vs CUDA vs CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"Using Apple M2 GPU (MPS) for training.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using NVIDIA CUDA for training.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for training.")
        
        self.model = AlphaZeroHexNet(board_size=self.board_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.best_model_path = "agents/DeepLearningAgent/best_model.pth"
        
        # Load existing weights to start
        if os.path.exists(self.best_model_path):
            print("Loading existing best weights...")
            try:
                self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            except:
                print("Weights corrupted! Starting fresh.")

    def execute_episode(self):
        """
        Self-Play: Generates training data.
        """
        board = Board(board_size=self.board_size)
        mcts = MCTS(self.model, cpu_ct=1.0)
        train_examples = []
        current_colour = Colour.RED
        
        while True:
            # temp=1 for exploration during self-play
            probs = mcts.get_action_prob(board, temp=1)
            
            canonical_board = encode_board(board, current_colour)
            train_examples.append([canonical_board, probs, current_colour])
            
            action_index = np.random.choice(len(probs), p=probs)
            row = action_index // self.board_size
            col = action_index % self.board_size
            
            board.set_tile_colour(row, col, current_colour)
            
            if board.has_ended(current_colour):
                winner = current_colour
                break
                
            current_colour = Colour.BLUE if current_colour == Colour.RED else Colour.RED

        final_data = []
        for x in train_examples:
            board_state, policy, player = x
            value = 1 if player == winner else -1
            final_data.append((board_state, policy, value))
            
        return final_data

    def evaluate_model(self, num_games=10):
        """
        ARENA MODE:
        Pits the current 'Candidate' model (self.model) against the 
        previous 'Champion' (loaded from disk).
        Returns True if Candidate wins >= 50% of games.
        """
        # 1. Load the Champion
        if not os.path.exists(self.best_model_path):
            print("No existing champion. Automatic win.")
            return True

        champion_net = AlphaZeroHexNet(self.board_size).to(self.device)
        champion_net.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
        champion_net.eval()
        
        challenger_net = self.model
        challenger_net.eval()
        
        wins = 0
        draws = 0 # Hex has no draws, but good to have variable logic
        
        print(f"ARENA ({num_games} games):", end=" ", flush=True)
        
        for i in range(num_games):
            # Challenger plays RED in even games, BLUE in odd games
            if i % 2 == 0:
                p1_net = challenger_net # RED
                p2_net = champion_net   # BLUE
                challenger_colour = Colour.RED
            else:
                p1_net = champion_net   # RED
                p2_net = challenger_net # BLUE
                challenger_colour = Colour.BLUE
            
            # Create fresh MCTS for every game
            mcts_p1 = MCTS(p1_net, cpu_ct=1.0)
            mcts_p2 = MCTS(p2_net, cpu_ct=1.0)
            
            board = Board(self.board_size)
            turn_colour = Colour.RED
            
            # Play one game
            while True:
                if turn_colour == Colour.RED:
                    # Temp=0 for competitive play (Greedy)
                    probs = mcts_p1.get_action_prob(board, temp=0)
                else:
                    probs = mcts_p2.get_action_prob(board, temp=0)
                
                best_move = np.argmax(probs)
                row = best_move // self.board_size
                col = best_move % self.board_size
                
                board.set_tile_colour(row, col, turn_colour)
                
                if board.has_ended(turn_colour):
                    winner = turn_colour
                    break
                    
                turn_colour = Colour.BLUE if turn_colour == Colour.RED else Colour.RED
            
            # Did challenger win?
            if winner == challenger_colour:
                wins += 1
                print("W", end="", flush=True)
            else:
                print("L", end="", flush=True)
        
        print(f" | Result: {wins}/{num_games}")
        
        # Threshold: Win 60% (6 out of 10) to replace champion
        win_rate = wins / num_games
        return win_rate >= 0.5

    def save_model_safely(self):
        """Atomic save to prevent corruption"""
        temp_path = "agents/DeepLearningAgent/temp_model.pth"
        torch.save(self.model.state_dict(), temp_path)
        os.replace(temp_path, self.best_model_path)
        print(f"SAVED NEW BEST MODEL to {self.best_model_path}")

    # def train(self, num_iterations=1000, episodes_per_iter=5):
    #     for i in range(num_iterations):
    #         print(f"\n=== Iteration {i+1} ===")
            
    #         # 1. SELF PLAY
    #         iteration_examples = []
    #         print("Self-Playing...", end=" ", flush=True)
    #         for e in range(episodes_per_iter):
    #             iteration_examples.extend(self.execute_episode())
    #             print(".", end="", flush=True)
    #         print(" Done!")
            
    #         if not iteration_examples: continue
    def train(self, num_iterations=1000, episodes_per_iter=50):
        # 1. SETUP MULTIPROCESSING
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        # Create a temp file path for sharing weights safely
        worker_model_path = "agents/DeepLearningAgent/worker_temp.pth"

        for i in range(num_iterations):
            print(f"\n=== Iteration {i+1} ===")
            
            # 2. SAVE WEIGHTS TO DISK (The Safe Handoff)
            torch.save(self.model.state_dict(), worker_model_path)
            
            # 3. PREPARE TASKS
            dev_name = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Pass the FILENAME, not the weights dict
            task_args = (worker_model_path, self.board_size, dev_name, 25)
            tasks = [task_args for _ in range(episodes_per_iter)]
            
            iteration_examples = []
            print(f"Self-Playing (Parallel)...", end=" ", flush=True)
            
            # 3. RUN WORKERS
            num_workers = 10
            
            with mp.Pool(processes=num_workers) as pool:
                # Use imap_unordered instead of map
                # This yields results AS SOON AS a worker finishes a game
                for game_data in pool.imap_unordered(self_play_worker, tasks):
                    iteration_examples.extend(game_data)
                    print(".", end="", flush=True)
            
            print(" Done!")

            if not iteration_examples: continue

            # 2. TRAIN
            boards, policies, values = zip(*iteration_examples)
            boards = torch.tensor(np.array(boards), dtype=torch.float32).to(self.device)
            policies = torch.tensor(np.array(policies), dtype=torch.float32).to(self.device)
            values = torch.tensor(np.array(values), dtype=torch.float32).to(self.device)
            
            dataset = TensorDataset(boards, policies, values)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            self.model.train()
            total_loss = 0
            for b, p_target, v_target in dataloader:
                self.optimizer.zero_grad()
                p_pred, v_pred = self.model(b)
                
                loss_v = torch.mean((v_target - v_pred.view(-1)) ** 2)
                loss_p = -torch.sum(p_target * p_pred) / p_target.size(0)
                loss = loss_v + loss_p
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Training Loss: {total_loss / len(dataloader):.4f}")
            
            # 3. ARENA EVALUATION
            # We skip arena for the very first iteration if no model exists
            print("Evaluating...", end=" ")
            if self.evaluate_model(num_games=10):
                print(">>> CHALLENGER WON! Accepting new weights.")
                self.save_model_safely()
            else:
                print(">>> CHALLENGER LOST. Discarding weights.")
                # Reload the previous best weights to forget the bad training
                self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))




if __name__ == "__main__":
    # Start loop
    # NOTE: Set board_size=11 for the real run!
    # Decrease episodes_per_iter if it's too slow on CPU
    # trainer = AlphaZeroTrainer(board_size=6)
    trainer = AlphaZeroTrainer(board_size=11)
    # trainer.train(num_iterations=1000, episodes_per_iter=100)
    trainer.train(num_iterations=1000, episodes_per_iter=50)
    # increasing or decreasing episodes per iter increases/decreases parrallelism. Use high values for gpu usage.
    # trainer.train(num_iterations=1000, episodes_per_iter=10)