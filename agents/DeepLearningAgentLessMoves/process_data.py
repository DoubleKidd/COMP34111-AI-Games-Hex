import os
import glob
import re
import numpy as np
import torch
from tqdm import tqdm

# --- IMPORT YOUR CLASSES ---
# Adjust these imports if your folder structure is different
from src.Board import Board
from src.Colour import Colour
from agents.DeepLearningAgent.utils import encode_board

# --- CONFIGURATION ---
BOARD_SIZE = 11
# Input folder containing .sgf or .sgfs files
INPUT_FOLDER = "agents/DeepLearningAgent/sl_raw_data"
# Output file path (I added .pt extension for PyTorch compatibility)
OUTPUT_FILE = "agents/DeepLearningAgent/sl_preprocessed_data.pt"

def parse_sgf_coord(coord_str, size):
    """
    Parses SGF coordinates (e.g., "aa", "ck").
    Returns (row, col) or None if invalid.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(coord_str) < 2: return None
    
    c_char = coord_str[0]
    r_char = coord_str[1]
    
    try:
        col = alphabet.index(c_char.lower())
        row = alphabet.index(r_char.lower())
    except ValueError:
        return None
    
    if col >= size or row >= size: return None
    return row, col

def process_sgf_files():
    # 1. Find all SGF and SGFS files recursively
    # This looks inside sl_raw_data and all its subfolders
    files = glob.glob(os.path.join(INPUT_FOLDER, "**", "*.sgf"), recursive=True)
    files += glob.glob(os.path.join(INPUT_FOLDER, "**", "*.sgfs"), recursive=True)
    
    print(f"Found {len(files)} log files in '{INPUT_FOLDER}'. Processing...")

    # These lists will hold data from ALL files combined
    all_boards = []
    all_policies = []
    all_values = []
    
    games_processed = 0
    skipped_games = 0

    # 2. Iterate through every file found
    for f_path in tqdm(files):
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            print(f"Skipping file {f_path}: {e}")
            continue

        # 3. Extract individual games from the file
        # SGF files group games in parenthesis: (;GM[1]...)(;GM[1]...)
        raw_games = re.findall(r'\((.*?)\)', file_content, re.DOTALL)

        for game_content in raw_games:
            # A. Determine Winner
            # RE[B+...] = Black (Red) Won, RE[W+...] = White (Blue) Won
            winner_match = re.search(r'RE\[([BW])', game_content)
            if not winner_match: 
                skipped_games += 1
                continue
            
            winner_colour = Colour.RED if winner_match.group(1) == 'B' else Colour.BLUE

            # B. Extract Moves
            # Regex for moves like ;B[aa] or ;W[cd]
            moves = re.findall(r';([BW])\[([a-zA-Z0-9]+)\]', game_content)
            
            # Skip empty or extremely short games
            if len(moves) < 5: 
                skipped_games += 1
                continue 

            # C. Replay the Game to generate tensors
            board = Board(BOARD_SIZE)
            current_colour = Colour.RED 
            
            valid_game = True
            game_boards = []
            game_policies = []
            game_values = []

            for player_char, coord_str in moves:
                # Optional: Check if player matches turn (B should be Red's turn)
                # expected_char = 'B' if current_colour == Colour.RED else 'W'
                # if player_char != expected_char: ... (Usually safe to ignore for KataGo selfplay)

                try:
                    row, col = parse_sgf_coord(coord_str, BOARD_SIZE)
                    if row is None: 
                        valid_game = False
                        break
                except: 
                    valid_game = False
                    break

                # --- 1. INPUT TENSOR (Canonical View, always see red) ---
                input_tensor = encode_board(board, current_colour)
                
                # --- 2. TARGET POLICY (The Move Pro Played) ---
                policy_target = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                
                # Handle Blue's perspective (Transpose Board -> Transpose Move)
                if current_colour == Colour.BLUE:
                    move_idx = col * BOARD_SIZE + row
                else:
                    move_idx = row * BOARD_SIZE + col
                    
                policy_target[move_idx] = 1.0
                
                # --- 3. TARGET VALUE (Did THIS player win?) ---
                outcome = 1.0 if current_colour == winner_colour else -1.0
                
                game_boards.append(input_tensor)
                game_policies.append(policy_target)
                game_values.append(outcome)
                
                # Apply move to board
                board.set_tile_colour(row, col, current_colour)
                current_colour = Colour.BLUE if current_colour == Colour.RED else Colour.RED
            
            if valid_game:
                all_boards.extend(game_boards)
                all_policies.extend(game_policies)
                all_values.extend(game_values)
                games_processed += 1
            else:
                skipped_games += 1

    print(f"\nProcessing Complete:")
    print(f" - Games Processed: {games_processed}")
    print(f" - Games Skipped:   {skipped_games}")
    print(f" - Total Positions: {len(all_boards)}")
    
    if len(all_boards) == 0:
        print("WARNING: No valid positions generated. Check input folder.")
        return

    # 4. Save Combined Data
    print(f"Saving to {OUTPUT_FILE}...")
    tensor_data = (
        torch.tensor(np.array(all_boards), dtype=torch.float32),
        torch.tensor(np.array(all_policies), dtype=torch.float32),
        torch.tensor(np.array(all_values), dtype=torch.float32)
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    torch.save(tensor_data, OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    process_sgf_files()