# self_play.py
# Phase 3: Self-Play Orchestration
# Objective: Generate game data using MCTS and the current best network.

import chess
import torch
import numpy as np
from typing import List, Tuple, Optional
import random
from tqdm import tqdm # For progress bars

import config
from chess_env import ChessEnv
from model import AttentionChessNet
from mcts import MCTS, Node # MCTS class and Node for type hinting
from replay_buffer import TrainingExample # For type hinting game history structure

def play_game(
    best_nn: AttentionChessNet,
    chess_env_instance: ChessEnv, # Re-use instance for efficiency
    mcts_instance: MCTS,          # Re-use instance
    num_mcts_simulations: int = config.MCTS_SIMULATIONS_SELF_PLAY,
    C_puct: float = config.C_PUCT,
    dirichlet_alpha: float = config.DIRICHLET_ALPHA,
    dirichlet_epsilon: float = config.DIRICHLET_EPSILON,
    temperature_moves: int = config.TEMPERATURE_MOVES,
    max_game_moves: int = 300 # Add a max game length to prevent infinitely long games
) -> Optional[List[Tuple[torch.Tensor, np.ndarray, float]]]:
    """
    Plays a single game of chess using MCTS guided by the neural network.
    Stores (board_state_tensor, mcts_policy_target, player_turn_perspective_outcome) for each move.

    Returns:
        List of training examples: List[Tuple[torch.Tensor, np.ndarray, float]]
        Each tuple is (board_state_tensor, mcts_policy_target_numpy_array, game_outcome_for_player_at_that_state)
        Returns None if game generation fails or is interrupted.
    """
    chess_env_instance.reset() # Start a new game
    game_history: List[Tuple[torch.Tensor, np.ndarray, chess.Color]] = [] # Store (state, policy, player_color_at_state)
    move_count = 0

    while not chess_env_instance.is_game_over() and move_count < max_game_moves:
        current_board_state = chess_env_instance.board.copy()
        current_player_color = chess_env_instance.get_current_player_color()

        # Get board tensor from current player's perspective
        # This is crucial: the NN expects input from the perspective of the player whose turn it is.
        s_tensor = chess_env_instance.board_to_input_tensor(current_player_color)

        # Run MCTS simulations
        # The MCTS's run_simulations should handle Dirichlet noise at root internally for self-play.
        # The board state passed to MCTS is the canonical board. MCTS handles perspectives.
        chosen_move_obj, policy_pi, _ = mcts_instance.run_simulations(
            root_board_state=current_board_state, # MCTS will use current_board_state.turn
            num_simulations=num_mcts_simulations,
            C_puct=C_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )

        if chosen_move_obj is None:
            # This can happen if MCTS finds no valid moves (e.g., unexpected terminal state or MCTS error)
            # Or if the game is already over but the loop condition didn't catch it (should not happen)
            print(f"Warning: MCTS returned no move for board:\n{current_board_state}\nGame over: {current_board_state.is_game_over()}")
            # If game is indeed over, the loop will break. If not, this is an issue.
            if not current_board_state.is_game_over():
                # This is problematic. Maybe pick a random legal move?
                # For now, let's end the game generation here.
                return None 
            break # Break if game is over

        # Store (s_tensor, policy_pi, current_player_color)
        # policy_pi is a numpy array from MCTS
        game_history.append((s_tensor, policy_pi, current_player_color))

        # Select move based on temperature
        # The policy_pi from MCTS is based on visit counts.
        # If temperature_moves > 0, sample. Else, pick best.
        actual_move_to_play: Optional[chess.Move] = None
        if move_count < temperature_moves:
            # Temperature τ=1.0: sample proportionally to visit counts (policy_pi)
            # policy_pi should sum to 1.
            # We need to map policy_pi indices back to moves.
            # This is tricky if policy_pi is sparse or chess_env.policy_index_to_action is not perfect.
            
            # A robust way: MCTS root node children have moves and visit counts.
            # Reconstruct probabilities from root node children visits.
            # The MCTS function might need to return the root node for this.
            # For now, let's assume policy_pi is a distribution over the 4672 actions.
            # We need to select a *legal* move based on this distribution.
            
            legal_moves = chess_env_instance.get_legal_moves()
            if not legal_moves: break # Should be caught by is_game_over

            move_probs = np.zeros(len(legal_moves))
            for i, move in enumerate(legal_moves):
                idx = chess_env_instance.action_to_policy_index(move, current_board_state)
                if 0 <= idx < len(policy_pi):
                    move_probs[i] = policy_pi[idx]
            
            # Normalize move_probs if they don't sum to 1 (due to mapping issues or masking)
            prob_sum = np.sum(move_probs)
            if prob_sum > 1e-6: # Avoid division by zero
                move_probs /= prob_sum
            else: # Fallback: uniform if all mapped probs are zero
                move_probs = np.ones(len(legal_moves)) / len(legal_moves)

            try:
                actual_move_to_play = np.random.choice(legal_moves, p=move_probs)
            except ValueError as e: # If move_probs doesn't sum to 1 or other issues
                print(f"Error sampling move with temperature: {e}. Board:\n{current_board_state}\nProbs: {move_probs}. Choosing random legal move.")
                actual_move_to_play = random.choice(legal_moves)

        else:
            # Temperature τ→0: Choose move with highest visit count (already given by MCTS as chosen_move_obj)
            actual_move_to_play = chosen_move_obj
        
        if actual_move_to_play is None: # Should not happen if legal moves exist
            print(f"Warning: actual_move_to_play is None. Board:\n{current_board_state}")
            legal_moves = chess_env_instance.get_legal_moves()
            if legal_moves: actual_move_to_play = random.choice(legal_moves) # Fallback
            else: break # No legal moves

        chess_env_instance.make_move(actual_move_to_play)
        move_count += 1

    # Game is over, get final outcome z
    # chess_env.get_game_outcome() is from perspective of current player to move.
    # If White won, outcome is 1 if White's turn, -1 if Black's turn.
    # We need the absolute outcome (e.g. 1 if White won, -1 if Black won, 0 for draw)
    
    final_board_state = chess_env_instance.board
    outcome_obj = final_board_state.outcome()
    
    z_absolute: float # 1 for White win, -1 for Black win, 0 for draw
    if outcome_obj is None: # Game ended due to max_game_moves
        # Treat as a draw, or use a heuristic evaluation if available
        z_absolute = 0.0 
        # print(f"Game ended due to max_game_moves ({max_game_moves}). Outcome: Draw (0.0)")
    elif outcome_obj.winner == chess.WHITE:
        z_absolute = 1.0
    elif outcome_obj.winner == chess.BLACK:
        z_absolute = -1.0
    else: # Draw
        z_absolute = 0.0

    # Propagate z back to all stored states for training targets
    # The training target value v for a state (s, π, col) is:
    # z_absolute if col == White
    # -z_absolute if col == Black
    # (assuming z_absolute is 1 for White win, -1 for Black win)
    
    training_data: List[Tuple[torch.Tensor, np.ndarray, float]] = []
    for s_tensor, policy_pi, player_color_at_state in game_history:
        value_target = 0.0
        if player_color_at_state == chess.WHITE:
            value_target = z_absolute
        else: # player_color_at_state == chess.BLACK
            value_target = -z_absolute
        
        training_data.append((s_tensor, policy_pi, value_target))
        
    return training_data

# --- Parallel Game Generation (Conceptual) ---
# Consider torch.multiprocessing or concurrent.futures for parallel generation.
# Each worker would need its own ChessEnv, MCTS instance, and a copy of the network weights.
# This is more advanced and requires careful management of resources and data aggregation.

# --- Testing ---
if __name__ == '__main__':
    print("Self-Play Testing (Conceptual - Requires Model, ChessEnv, MCTS)")

    if not torch.cuda.is_available():
        print("CUDA not available, tests will run on CPU.")
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda")

    try:
        # Initialize components (potentially with smaller configs for speed if testing on CPU)
        chess_env = ChessEnv()
        
        # Use a model that can run on the available device
        # For quick testing, a smaller model might be preferred if not on powerful GPU.
        # Let's assume config values are set for a reasonable test.
        best_nn_model = AttentionChessNet().to(DEVICE)
        best_nn_model.eval() # Ensure it's in eval mode for self-play inference

        mcts_algo = MCTS(chess_env, best_nn_model, device=DEVICE)

        print("\nStarting a single self-play game...")
        # Use fewer MCTS simulations for faster testing
        test_mcts_sims = config.MCTS_SIMULATIONS_SELF_PLAY // 10 if config.MCTS_SIMULATIONS_SELF_PLAY > 20 else 20 # e.g. 20-80
        
        game_data = play_game(
            best_nn=best_nn_model,
            chess_env_instance=chess_env, # Pass the instance
            mcts_instance=mcts_algo,       # Pass the instance
            num_mcts_simulations=max(16, test_mcts_sims//2), # Reduced for faster test
            temperature_moves=15 # Shorter period of random moves for test
        )

        if game_data:
            print(f"\nSelf-play game completed. Generated {len(game_data)} training examples.")
            first_example_state, first_example_policy, first_example_value = game_data[0]
            print(f"First example - state tensor shape: {first_example_state.shape}")
            print(f"First example - policy target shape: {first_example_policy.shape}") # Numpy array
            print(f"First example - game outcome for player: {first_example_value}")

            # Check types
            assert isinstance(first_example_state, torch.Tensor)
            assert isinstance(first_example_policy, np.ndarray)
            assert isinstance(first_example_value, float)
            assert first_example_policy.shape == (config.POLICY_OUTPUT_SIZE,)

        else:
            print("\nSelf-play game failed to generate data.")

        print("\nSelf-Play conceptual test finished.")
        print("NOTE: This test relies on functional MCTS, Model, and ChessEnv.")
        print("Issues in underlying modules (especially move mapping in ChessEnv or MCTS logic) will affect self-play.")

    except Exception as e:
        print(f"An error occurred during self-play testing: {e}")
        import traceback
        traceback.print_exc()