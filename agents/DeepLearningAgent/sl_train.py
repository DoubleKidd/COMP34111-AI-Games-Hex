import os
import shutil
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader

# --- IMPORT YOUR GAME CLASSES ---
from agents.DeepLearningAgent.model import AlphaZeroHexNet
from agents.DeepLearningAgent.mcts import MCTS
from src.Board import Board
from src.Colour import Colour

# --- CONFIGURATION ---
DATA_PATH = "agents/DeepLearningAgent/sl_preprocessed_data.pt"
MODEL_PATH = "agents/DeepLearningAgent/best_model.pth"
HISTORY_DIR = "agents/DeepLearningAgent/history/"
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

# ARENA SETTINGS
ARENA_GAMES = 20        # Number of games to play for evaluation
ARENA_SIMULATIONS = 25  # Lower simulations for speed during check
ARENA_CPU_CT = 1.0      # Exploration constant for arena

def archive_current_model():
    """Archives the current champion to history."""
    if os.path.exists(MODEL_PATH):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"model_sl_backup_{timestamp}.pth"
        backup_path = os.path.join(HISTORY_DIR, backup_name)
        print(f"Archiving current champion to {backup_path}...")
        shutil.copy(MODEL_PATH, backup_path)

def play_arena_game(args):
    """
    Worker function to play one game between Candidate (New) and Champion (Old).
    """
    candidate_state, champion_state, game_id, board_size, device_name = args
    
    # Setup Device
    device = torch.device(device_name)
    
    # Load Models
    candidate_net = AlphaZeroHexNet(board_size).to(device)
    candidate_net.load_state_dict(candidate_state)
    candidate_net.eval()
    
    champion_net = AlphaZeroHexNet(board_size).to(device)
    champion_net.load_state_dict(champion_state)
    champion_net.eval()
    
    # Determine Colors: Candidate is RED in even games, BLUE in odd
    if game_id % 2 == 0:
        p1_net = candidate_net # RED
        p2_net = champion_net   # BLUE
        candidate_colour = Colour.RED
    else:
        p1_net = champion_net   # RED
        p2_net = candidate_net # BLUE
        candidate_colour = Colour.BLUE
        
    mcts_p1 = MCTS(p1_net, cpu_ct=ARENA_CPU_CT)
    mcts_p2 = MCTS(p2_net, cpu_ct=ARENA_CPU_CT)
    
    board = Board(board_size)
    turn_colour = Colour.RED
    
    while True:
        # Get move probabilities
        if turn_colour == Colour.RED:
            probs = mcts_p1.get_action_prob(board, temp=0.1, simulations=ARENA_SIMULATIONS)
        else:
            probs = mcts_p2.get_action_prob(board, temp=0.1, simulations=ARENA_SIMULATIONS)
            
        best_move = np.argmax(probs)
        row = best_move // board_size
        col = best_move % board_size
        
        board.set_tile_colour(row, col, turn_colour)
        
        if board.has_ended(turn_colour):
            return 1 if turn_colour == candidate_colour else 0
            
        turn_colour = Colour.BLUE if turn_colour == Colour.RED else Colour.RED

def run_arena_evaluation(new_model_state, board_size=11):
    """
    Pits the new model state against the saved 'best_model.pth'.
    Returns True if the new model wins >= 55% of games.
    """
    if not os.path.exists(MODEL_PATH):
        print("No existing champion found. Automatic promotion.")
        return True
        
    print(f"\n=== ARENA EVALUATION ({ARENA_GAMES} Games) ===")
    print("Candidate (New) vs Champion (Old)...")
    
    # Load Champion State
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For multiprocessing, we pass state dicts (CPU) to avoid pickling issues
    champion_state = torch.load(MODEL_PATH, map_location="cpu")
    
    # Prepare arguments for workers
    # We use CPU or CUDA string for device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    tasks = []
    for i in range(ARENA_GAMES):
        tasks.append((new_model_state, champion_state, i, board_size, device_name))
    
    wins = 0
    # Use roughly # of cores
    num_workers = min(ARENA_GAMES, os.cpu_count())
    
    # MP Context for compatibility
    ctx = mp.get_context("spawn")
    
    with ctx.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(play_arena_game, tasks):
            wins += result
            print("W" if result == 1 else "L", end="", flush=True)
            
    win_rate = wins / ARENA_GAMES
    print(f"\nArena Result: {wins}/{ARENA_GAMES} ({win_rate*100:.1f}%)")
    
    # Threshold: 55% win rate required to replace
    return win_rate >= 0.55

def train_supervised():
    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Training on {device}...")

    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file {DATA_PATH} not found.")
        print("Please run 'process_data.py' first.")
        return

    print(f"Loading dataset from {DATA_PATH}...")
    try:
        boards, policies, values = torch.load(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    dataset = TensorDataset(boards, policies, values)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} training positions.")
    
    # 3. Initialize Model
    model = AlphaZeroHexNet(board_size=11).to(device)
    
    # OPTIONAL: Start from existing weights?
    # Usually for SL we start fresh or from previous SL checkpoint, 
    # but here we likely start fresh to "learn from pro" without bias.
    # If you want to refine, uncomment below:
    # if os.path.exists(MODEL_PATH):
    #     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 4. Training Loop
    model.train()
    
    print("\n=== STARTING SUPERVISED TRAINING ===")
    for epoch in range(EPOCHS):
        total_loss = 0
        total_p_loss = 0
        total_v_loss = 0
        
        for b, p_target, v_target in dataloader:
            b, p_target, v_target = b.to(device), p_target.to(device), v_target.to(device)
            
            optimizer.zero_grad()
            p_pred, v_pred = model(b)
            
            # Policy Loss
            loss_p = -torch.sum(p_target * p_pred) / p_target.size(0)
            
            # Value Loss
            loss_v = torch.mean((v_target - v_pred.view(-1)) ** 2)
            
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_p_loss += loss_p.item()
            total_v_loss += loss_v.item()
            
        # Print stats
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} (Pol: {total_p_loss/len(dataloader):.4f}, Val: {total_v_loss/len(dataloader):.4f})")

    # 5. Evaluate against Old Model
    # Move model to CPU to extract state_dict safely for multiprocessing
    model.to("cpu")
    new_state = copy.deepcopy(model.state_dict())
    
    if run_arena_evaluation(new_state):
        print(">>> NEW MODEL IS SUPERIOR! Saving...")
        archive_current_model()
        torch.save(new_state, MODEL_PATH)
        print(f"Saved new champion to {MODEL_PATH}")
    else:
        print(">>> NEW MODEL FAILED TO BEAT CHAMPION. Discarding.")

if __name__ == "__main__":
    # Fix for MP on some platforms
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    train_supervised()