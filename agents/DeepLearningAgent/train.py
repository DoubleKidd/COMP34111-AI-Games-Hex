import os
os.environ["XBYAK_ARCH_AARCH64"] = "0"

# --- CRITICAL PERFORMANCE FIX ---
# Force Numpy/PyTorch to use only 1 thread per process.
# This prevents "Thread Thrashing" when we run multiple workers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
# ... rest of your imports ...


import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
import copy
import time
import random
import glob

# This forces PyTorch to use generic math functions instead of 
# trying to probe the ARM processor hardware (which crashes Docker).
torch.backends.mkldnn.enabled = False
torch.backends.nnpack.enabled = False

from src.Board import Board
from src.Colour import Colour
from agents.DeepLearningAgent.model import AlphaZeroHexNet
from agents.DeepLearningAgent.mcts import MCTS
from agents.DeepLearningAgent.utils import encode_board

# --- STANDALONE ARENA WORKER ---
def arena_worker(args):
    """
    Independent worker that plays a single Arena game.
    """
    (challenger_path, opponent_path, board_size, device_name, 
     simulations, cpu_ct, game_index) = args

    try:
        device = torch.device(device_name)
        
        # 1. Load Challenger (The model we are training)
        challenger_net = AlphaZeroHexNet(board_size).to(device)
        challenger_net.load_state_dict(torch.load(challenger_path, map_location=device, weights_only=True))
        challenger_net.eval()
        
        # 2. Load Opponent (Champion or Ghost)
        opponent_net = AlphaZeroHexNet(board_size).to(device)
        opponent_net.load_state_dict(torch.load(opponent_path, map_location=device, weights_only=True))
        opponent_net.eval()

        # 3. Determine Colours (Challenger is RED in even games, BLUE in odd)
        if game_index % 2 == 0:
            p1_net = challenger_net # RED
            p2_net = opponent_net   # BLUE
            challenger_colour = Colour.RED
        else:
            p1_net = opponent_net   # RED
            p2_net = challenger_net # BLUE
            challenger_colour = Colour.BLUE

        # 4. Play Game
        board = Board(board_size)
        mcts_p1 = MCTS(p1_net, cpu_ct=cpu_ct)
        mcts_p2 = MCTS(p2_net, cpu_ct=cpu_ct)
        
        turn_colour = Colour.RED
        
        while True:
            # Low temp for Arena (competitive)
            if turn_colour == Colour.RED:
                probs = mcts_p1.get_action_prob(board, temp=0.1, simulations=simulations)
            else:
                probs = mcts_p2.get_action_prob(board, temp=0.1, simulations=simulations)
            
            # Deterministic move selection for evaluation
            best_move = np.argmax(probs)
            row = best_move // board_size
            col = best_move % board_size
            
            board.set_tile_colour(row, col, turn_colour)
            
            if board.has_ended(turn_colour):
                winner = turn_colour
                break
                
            turn_colour = Colour.BLUE if turn_colour == Colour.RED else Colour.RED
        
        # Return 1 if Challenger won, 0 otherwise
        return 1 if winner == challenger_colour else 0

    except Exception as e:
        print(f"Arena Error: {e}")
        return 0 # Assume loss on error

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
    model_path, board_size, device_name, simulations, cpu_ct, temp = args
    
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
        mcts = MCTS(local_model, cpu_ct=cpu_ct)
        
        # 3. Play Game
        board = Board(board_size=board_size)
        train_examples = []
        current_colour = Colour.RED
        
        while True:
            # Get action probabilities
            real_probs = mcts.get_action_prob(board, temp=temp, simulations=simulations) # The real view

            # 2. Create a copy for the TRAINING DATA
            train_probs = list(real_probs)

            # If it's Blue's turn, the MCTS returned "Real Board" probabilities (Horizontal).
            # But we are saving a "Canonical Board" (Vertical).
            # We must transpose the probs to match the board.
            if current_colour == Colour.BLUE:
                # Convert list to numpy, reshape, transpose, flatten, convert back to list
                probs_np = np.array(real_probs)
                probs_reshaped = probs_np.reshape(board_size, board_size)
                train_probs = probs_reshaped.T.flatten().tolist()
            
            canonical_board = encode_board(board, current_colour)
            train_examples.append([canonical_board, train_probs, current_colour]) # the always vertical view
            
            action_index = np.random.choice(len(real_probs), p=real_probs)
            row = action_index // board_size
            col = action_index % board_size
            
            board.set_tile_colour(row, col, current_colour)
            
            if board.has_ended(current_colour):
                winner = current_colour
                break
                
            current_colour = Colour.BLUE if current_colour == Colour.RED else Colour.RED

        # 4. Format Data
        final_data = []
        total_moves = len(train_examples)
        
        # A decay factor of 0.99 significantly penalizes long games.
        # Example: Win in 1 move = 0.99 value. Win in 60 moves = 0.54 value.
        gamma = 0.99 

        for i, x in enumerate(train_examples):
            board_state, policy, player = x
            
            # Raw Win/Loss
            raw_value = 1 if player == winner else -1
            
            # Calculate moves remaining from THIS state to the end of the game
            moves_until_end = total_moves - i
            
            # Apply decay: 
            # If winning (1): Value drops from 1.0 -> 0.0 as game gets longer.
            # If losing (-1): Value rises from -1.0 -> 0.0 as game gets longer (prolonging defeat).
            time_weighted_value = raw_value * (gamma ** (moves_until_end-board_size))
            
            final_data.append((board_state, policy, time_weighted_value))

        # # 4. Format Data
        # final_data = []
        # for x in train_examples:
        #     board_state, policy, player = x
        #     value = 1 if player == winner else -1
        #     final_data.append((board_state, policy, value))
            
        return final_data

    except Exception as e:
        # This catches the error so the main process doesn't hang
        print(f"Worker Error: {e}")
        return []

class AlphaZeroTrainer:
    def __init__(self, board_size=11, simulations=50, cpu_ct=0.5, temp=0.1, evaluation_simulations=100, evaluation_cpu_ct=1.0):
        self.simulations = simulations
        self.board_size = board_size
        self.cpu_ct = cpu_ct
        self.temp = temp
        self.evaluation_simulations = evaluation_simulations
        self.evaluation_cpu_ct = evaluation_cpu_ct
        
        # Detect Mac M2 (MPS) vs CUDA vs CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"Using Apple M2 GPU (MPS) for training.")
        # elif torch.cuda.is_available(): # CPU was found to be faster.
        #     self.device = torch.device("cuda")
        #     print("Using NVIDIA CUDA for training.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for training.")
        
        # INITIALIZE MODEL
        self.model = AlphaZeroHexNet(board_size=self.board_size).to(self.device)

        # JIT TRACE SPEED OPTIMIZATION
        # This compiles the PyTorch model into optimized C++ instructions
        # Only works on CPU or CUDA (Not MPS/Mac yet)
        if self.device.type == 'cpu' or self.device.type == 'cuda':
            try:
                print("Optimizing model with JIT Trace...", end=" ")
                self.model.eval()
                # Create a dummy input (Batch=1, Channels=2, H=Size, W=Size)
                example_input = torch.randn(1, 2, self.board_size, self.board_size).to(self.device)
                
                # Trace the model to compile it to C++
                self.model = torch.jit.trace(self.model, example_input)
                print("Done!")
            except Exception as e:
                # If optimization fails, we just print the error and continue with the standard model
                print(f"JIT Trace failed (ignoring): {e}")
        # Add 'weight_decay=1e-4' (Standard value for AlphaZero)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        self.best_model_path = "agents/DeepLearningAgent/best_model.pth"

        # Create a directory for the Hall of Fame models
        self.history_path = "agents/DeepLearningAgent/history/"
        os.makedirs(self.history_path, exist_ok=True)

        # NEW: Determine where to start counting
        self.start_iteration = self._get_next_iteration_number()
        print(f"Resuming training from Iteration {self.start_iteration}...")
        
        # Load existing weights to start
        if os.path.exists(self.best_model_path):
            print("Loading existing best weights...")
            try:
                self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            except:
                print("Weights corrupted! Starting fresh.")

    def _get_next_iteration_number(self):
        """
        Scans the history folder for 'model_X.pth' files
        and returns the next available number (X + 1).
        """
        # Find all files matching "model_*.pth"
        files = glob.glob(os.path.join(self.history_path, "model_*.pth"))
        
        if not files:
            return 1  # No history? Start at 1
            
        # Regex to extract the number from "agents/.../model_12.pth"
        # We look for digits (\d+) right before .pth
        versions = []
        for f in files:
            match = re.search(r"model_(\d+).pth", f)
            if match:
                versions.append(int(match.group(1)))
        
        if not versions:
            return 1
            
        return max(versions) + 1

    def execute_episode(self):
        """
        Self-Play: Generates training data.
        """
        board = Board(board_size=self.board_size)
        mcts = MCTS(self.model, cpu_ct=self.cpu_ct)
        train_examples = []
        current_colour = Colour.RED
        
        while True:
            real_probs = mcts.get_action_prob(board, temp=self.temp, simulations=self.simulations)

            # 2. Create a copy for the TRAINING DATA
            train_probs = list(real_probs)

            # If blue, flip board.
            # If it's Blue's turn, the MCTS returned "Real Board" probabilities (Horizontal).
            # But we are saving a "Canonical Board" (Vertical).
            # We must transpose the probs to match the board.
            if current_colour == Colour.BLUE:
                # Convert list to numpy, reshape, transpose, flatten, convert back to list
                probs_np = np.array(real_probs)
                probs_reshaped = probs_np.reshape(self.board_size, self.board_size)
                train_probs = probs_reshaped.T.flatten().tolist()
            
            
            canonical_board = encode_board(board, current_colour)
            train_examples.append([canonical_board, train_probs, current_colour])
            
            action_index = np.random.choice(len(real_probs), p=real_probs)
            row = action_index // self.board_size
            col = action_index % self.board_size
            
            board.set_tile_colour(row, col, current_colour)
            
            if board.has_ended(current_colour):
                winner = current_colour
                break
                
            current_colour = Colour.BLUE if current_colour == Colour.RED else Colour.RED

        # 4. Format Data
        final_data = []
        total_moves = len(train_examples)
        
        # A decay factor of 0.99 significantly penalizes long games.
        # Example: Win in 1 move = 0.99 value. Win in 60 moves = 0.54 value.
        gamma = 0.99 

        for i, x in enumerate(train_examples):
            board_state, policy, player = x
            
            # Raw Win/Loss
            raw_value = 1 if player == winner else -1
            
            # Calculate moves remaining from THIS state to the end of the game
            moves_until_end = total_moves - i
            
            # Apply decay: 
            # If winning (1): Value drops from 1.0 -> 0.0 as game gets longer.
            # If losing (-1): Value rises from -1.0 -> 0.0 as game gets longer (prolonging defeat).
            time_weighted_value = raw_value * (gamma ** moves_until_end)
            
            final_data.append((board_state, policy, time_weighted_value))

        # final_data = []
        # for x in train_examples:
        #     board_state, policy, player = x
        #     value = 1 if player == winner else -1
        #     final_data.append((board_state, policy, value))
            
        return final_data

    def evaluate_model(self, num_games=10):
        """
        PARALLEL ARENA MODE
        """
        if not os.path.exists(self.best_model_path):
            print("No existing champion. Automatic win.")
            return True

        # 1. Save the CURRENT (Challenger) model to a temp file for workers to load
        challenger_path = "agents/DeepLearningAgent/challenger_temp.pth"
        torch.save(self.model.state_dict(), challenger_path)
        
        # 2. Identify Potential Opponents
        past_models = glob.glob(os.path.join(self.history_path, "*.pth"))
        
        # 3. Prepare Tasks (Matchups)
        tasks = []
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"ARENA ({num_games} games):", end=" ", flush=True)

        for i in range(num_games):
            # Decide Opponent: 30% Ghost, 70% Champion
            if past_models and random.random() < 0.3:
                opponent_path = random.choice(past_models)
                # opp_name = "Ghost"
            else:
                opponent_path = self.best_model_path
                # opp_name = "Champion"

            # tuple arguments for arena_worker
            task = (
                challenger_path, 
                opponent_path, 
                self.board_size, 
                dev_name, 
                self.evaluation_simulations, 
                self.evaluation_cpu_ct, 
                i  # game_index (determines colour)
            )
            tasks.append(task)

        # 4. Run Parallel Arena
        wins = 0
        num_workers = os.cpu_count() # Use all cores
        
        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(arena_worker, tasks):
                wins += result
                
                # Visual feedback
                if result == 1:
                    print("W", end="", flush=True)
                else:
                    print("L", end="", flush=True)

        print(f" | Result: {wins}/{num_games}")
        
        # Cleanup temp file
        if os.path.exists(challenger_path):
            os.remove(challenger_path)

        # Require 60% win rate
        return (wins / num_games) >= 0.6

    # def evaluate_model(self, num_games=10):
    #     """
    #     ARENA MODE:
    #     Pits the Candidate against:
    #     1. The Current Champion (Primary test)
    #     2. A Random Historic Model (Sanity check against forgetting)
    #     """
    #     if not os.path.exists(self.best_model_path):
    #         print("No existing champion. Automatic win.")
    #         return True

    #     # --- OPPONENT SELECTION ---
    #     # We want to play most games against the Best, but some against the Past.
    #     # If we have history, pick a random past model.
    #     past_models = glob.glob(os.path.join(self.history_path, "*.pth"))
        
    #     # Load the Current Champion
    #     champion_net = AlphaZeroHexNet(self.board_size).to(self.device)
    #     champion_net.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
    #     champion_net.eval()
        
    #     # Load a Ghost (Historic) Model if available
    #     ghost_net = None
    #     if len(past_models) > 0:
    #         ghost_path = random.choice(past_models)
    #         ghost_net = AlphaZeroHexNet(self.board_size).to(self.device)
    #         ghost_net.load_state_dict(torch.load(ghost_path, map_location=self.device, weights_only=True))
    #         ghost_net.eval()
    #         print(f"Arena Opponents: Champion & Ghost ({os.path.basename(ghost_path)})")
    #     else:
    #         print("Arena Opponents: Champion Only")

    #     challenger_net = self.model
    #     challenger_net.eval()
        
    #     wins = 0
        
    #     print(f"ARENA ({num_games} games):", end=" ", flush=True)
        
    #     for i in range(num_games):
    #         # Decide Opponent: 30% chance to play Ghost, 70% Champion
    #         if ghost_net and random.random() < 0.3:
    #             opponent_net = ghost_net
    #             opp_name = "G" # Ghost
    #         else:
    #             opponent_net = champion_net
    #             opp_name = "C" # Champion

    #         # Challenger plays RED in even games, BLUE in odd games
    #         if i % 2 == 0:
    #             p1_net = challenger_net # RED
    #             p2_net = opponent_net   # BLUE
    #             challenger_colour = Colour.RED
    #         else:
    #             p1_net = opponent_net   # RED
    #             p2_net = challenger_net # BLUE
    #             challenger_colour = Colour.BLUE
            
    #         # MCTS Setup
    #         mcts_p1 = MCTS(p1_net, cpu_ct=self.evaluation_cpu_ct)
    #         mcts_p2 = MCTS(p2_net, cpu_ct=self.evaluation_cpu_ct)
            
    #         board = Board(self.board_size)
    #         turn_colour = Colour.RED
            
    #         while True:
    #             # Low temp for Arena (competitive)
    #             if turn_colour == Colour.RED:
    #                 probs = mcts_p1.get_action_prob(board, temp=0.1, simulations=self.evaluation_simulations)
    #             else:
    #                 probs = mcts_p2.get_action_prob(board, temp=0.1, simulations=self.evaluation_simulations)
                
    #             best_move = np.argmax(probs)
    #             row = best_move // self.board_size
    #             col = best_move % self.board_size
                
    #             board.set_tile_colour(row, col, turn_colour)
                
    #             if board.has_ended(turn_colour):
    #                 winner = turn_colour
    #                 break
                    
    #             turn_colour = Colour.BLUE if turn_colour == Colour.RED else Colour.RED
            
    #         if winner == challenger_colour:
    #             wins += 1
    #             print("W", end="", flush=True)
    #         else:
    #             print("L", end="", flush=True)
        
    #     print(f" | Result: {wins}/{num_games}")
        
    #     # Require 60% win rate if playing against a mixed pool (easier than pure Champion)
    #     return (wins / num_games) >= 0.6

    def save_model_safely(self, iteration_index):
        """Atomic save to best model, plus archiving to history."""
        temp_path = "agents/DeepLearningAgent/temp_model.pth"
        torch.save(self.model.state_dict(), temp_path)
        
        # 1. Update the Main Champion
        os.replace(temp_path, self.best_model_path)
        
        # 2. Add to Hall of Fame (Save a copy as model_1.pth, model_2.pth, etc.)
        history_name = os.path.join(self.history_path, f"model_{iteration_index}.pth")
        torch.save(self.model.state_dict(), history_name)
        
        print(f"SAVED NEW BEST MODEL to {self.best_model_path}")
        print(f"ARCHIVED to {history_name}")

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
    def train(self, num_iterations=1000, episodes_per_iter=50, epochs=10):
        # 1. SETUP MULTIPROCESSING
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        # Create a temp file path for sharing weights safely
        worker_model_path = "agents/DeepLearningAgent/worker_temp.pth"

        # CHANGED: Loop from start_iter -> start_iter + num_iters
        end_iteration = self.start_iteration + num_iterations
        
        for i in range(self.start_iteration, end_iteration):
            iter_start_time = time.time()   # <--- START TIMER
            print(f"\n=== Iteration {i} ===") # 'i' depends on the most recent historical best model iteration.
            
            # 2. SAVE WEIGHTS TO DISK (The Safe Handoff)
            torch.save(self.model.state_dict(), worker_model_path)
            
            # 3. PREPARE TASKS
            dev_name = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Pass the FILENAME, not the weights dict
            task_args = (worker_model_path, self.board_size, dev_name, self.simulations, self.cpu_ct, self.temp)
            tasks = [task_args for _ in range(episodes_per_iter)]
            
            iteration_examples = []
            print(f"Self-Playing (Parallel)...", end=" ", flush=True)
            
            # Currently running in serial as parallelisation with a windows generated tensorflow model breaks on mac.
            # num_workers = 1

            # 3. RUN WORKERS
            # Dynamically grab the number of cores available in the Docker container
            total_cores = os.cpu_count()
            
            # Recommendation: Use all cores. 
            # If you find your computer freezing, use: max(1, total_cores - 1)
            num_workers = total_cores
            
            with mp.Pool(processes=num_workers) as pool:
                # Use imap_unordered instead of map
                # This yields results AS SOON AS a worker finishes a game
                for game_data in pool.imap_unordered(self_play_worker, tasks):
                    iteration_examples.extend(game_data)
                    print(".", end="", flush=True)

            # 3. RUN WORKERS (SERIAL)
            # We execute the worker function directly in a loop.
            # This bypasses the multiprocessing/Docker crash completely.
            # for task in tasks:
            #     game_data = self_play_worker(task)
            #     iteration_examples.extend(game_data)
            #     print(".", end="", flush=True)
            
            print(" Done!")

            if not iteration_examples: continue

            # 2. TRAIN
            boards, policies, values = zip(*iteration_examples)
            boards = torch.tensor(np.array(boards), dtype=torch.float32).to(self.device)
            policies = torch.tensor(np.array(policies), dtype=torch.float32).to(self.device)
            values = torch.tensor(np.array(values), dtype=torch.float32).to(self.device)
            
            dataset = TensorDataset(boards, policies, values)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
            
            self.model.train()


            # Run X epochs
            for epoch in range(epochs): 
                total_loss = 0
                total_policy_loss = 0
                total_validation_loss = 0
                for b, p_target, v_target in dataloader:
                    self.optimizer.zero_grad()
                    p_pred, v_pred = self.model(b)
                    
                    loss_v = torch.mean((v_target - v_pred.view(-1)) ** 2)
                    loss_p = -torch.sum(p_target * p_pred) / p_target.size(0)
                    loss = loss_v + loss_p
                    total_policy_loss += loss_p
                    total_validation_loss += loss_v
                    
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            
            print(f"Training Policy Loss: {total_policy_loss / len(dataloader):.4f}")
            print(f"Training Validation Loss: {total_validation_loss / len(dataloader):.4f}")
            print(f"Training Loss: {total_loss / len(dataloader):.4f}")
            
            # 3. ARENA EVALUATION
            # We skip arena for the very first iteration if no model exists
            print("Evaluating...", end=" ")
            if self.evaluate_model(num_games=16):
                print(">>> CHALLENGER WON! Accepting new weights.")
                self.save_model_safely(i)
            else:
                print(">>> CHALLENGER LOST. Discarding weights.")
                # Reload the previous best weights to forget the bad training
                self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
            elapsed = time.time() - iter_start_time
            print(f"Iteration {i+1} took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")




if __name__ == "__main__":
    # Start loop
    # NOTE: Set board_size=11 for the real run!
    # Decrease episodes_per_iter if it's too slow on CPU
    # trainer = AlphaZeroTrainer(board_size=6)
    # increasing or decreasing episodes per iter increases/decreases parrallelism. Use high values for gpu usage.
    # trainer.train(num_iterations=1000, episodes_per_iter=10)

    # For test
    trainer = AlphaZeroTrainer(board_size=11, simulations=25, cpu_ct=0.8, temp=1.1)
    trainer.train(num_iterations=200, episodes_per_iter=32, epochs=5)
