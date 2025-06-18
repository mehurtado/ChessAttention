# evaluate.py
# Phase 5: Evaluation Framework
# Objective: Evaluate the strength of new network checkpoints.

import chess
import torch
import numpy as np
from typing import Tuple
from tqdm import tqdm # For progress bar

import config
from chess_env import ChessEnv
from model import AttentionChessNet
from mcts import MCTS

def play_eval_game(
    net1: AttentionChessNet, # e.g., current best
    net2: AttentionChessNet, # e.g., candidate
    chess_env_instance: ChessEnv, # Re-use instance
    device: torch.device,
    num_mcts_sims_eval: int = config.MCTS_SIMULATIONS_EVAL,
    C_puct: float = config.C_PUCT,
    max_game_moves: int = 300 # Max moves for an eval game
) -> Tuple[float, chess.Color | None]: # Returns outcome for net2 (1 win, 0 draw, -1 loss), and winner color
    """
    Plays a single evaluation game between two networks.
    net1 and net2 will have MCTS instances created for them.
    Alternates starting player implicitly if called multiple times with swapped nets.
    This function plays one game where net1 is white, net2 is black, or vice-versa,
    depending on who is assigned to play white/black.
    Let's define net1 plays as White, net2 plays as Black for this call.
    """
    chess_env_instance.reset()
    
    # MCTS instances for each network for this game
    # No Dirichlet noise and deterministic move selection (temp -> 0) for evaluation
    mcts1 = MCTS(chess_env_instance, net1, device)
    mcts2 = MCTS(chess_env_instance, net2, device)

    nets = {chess.WHITE: net1, chess.BLACK: net2}
    mcts_players = {chess.WHITE: mcts1, chess.BLACK: mcts2}
    
    move_count = 0
    while not chess_env_instance.is_game_over() and move_count < max_game_moves:
        current_player_color = chess_env_instance.get_current_player_color()
        current_mcts = mcts_players[current_player_color]
        current_board_state = chess_env_instance.board.copy()

        # Run MCTS simulations for the current player
        # For evaluation: no Dirichlet noise, deterministic move selection (highest visit count)
        chosen_move_obj, _, _ = current_mcts.run_simulations(
            root_board_state=current_board_state,
            num_simulations=num_mcts_sims_eval,
            C_puct=C_puct,
            dirichlet_alpha=0.0, # No noise for eval
            dirichlet_epsilon=0.0  # No noise for eval
        )

        if chosen_move_obj is None:
            # Should only happen if game is already over or no legal moves
            if not current_board_state.is_game_over():
                print(f"Warning: MCTS returned no move in eval game for player {current_player_color} on board:\n{current_board_state}")
            break 

        chess_env_instance.make_move(chosen_move_obj)
        move_count += 1

    # Determine game outcome
    outcome_obj = chess_env_instance.board.outcome()
    winner_color: Optional[chess.Color] = None
    
    # Outcome for net2 (which played as Black in this setup if net1=White, net2=Black)
    # Or, more generally, outcome for the network that was assigned to player2_color
    # Let's define player1_color and player2_color for clarity if nets are swapped outside.
    # For this function, assume net1 is P1, net2 is P2.
    # If P1=White, P2=Black:
    #   If White wins (P1 wins), net2 loses (-1).
    #   If Black wins (P2 wins), net2 wins (1).
    #   If Draw, net2 draws (0).

    # Let's simplify: this function determines outcome for the fixed assignment:
    # net1 plays White, net2 plays Black. We return outcome for net2.
    
    outcome_for_net2: float = 0.0
    if outcome_obj is None: # Max moves reached
        outcome_for_net2 = 0.0 # Draw
        winner_color = None
    elif outcome_obj.winner == chess.WHITE: # net1 (White) won
        outcome_for_net2 = -1.0 
        winner_color = chess.WHITE
    elif outcome_obj.winner == chess.BLACK: # net2 (Black) won
        outcome_for_net2 = 1.0
        winner_color = chess.BLACK
    else: # Draw
        outcome_for_net2 = 0.0
        winner_color = None
        
    return outcome_for_net2, winner_color


def evaluate_networks(
    net1_path: str, # Path to current best nn state_dict
    net2_path: str, # Path to candidate nn state_dict
    num_eval_games: int = config.NUM_EVAL_GAMES,
    num_mcts_sims_eval: int = config.MCTS_SIMULATIONS_EVAL,
    device_str: str = config.DEVICE
) -> float:
    """
    Evaluates two network checkpoints against each other.
    Loads models from paths.
    Plays num_eval_games, alternating starting player.
    Returns the win rate of net2 against net1.
    """
    DEVICE = torch.device(device_str)

    # Load networks
    # Assuming model architecture is fixed and defined by AttentionChessNet()
    # Adjust if model architecture parameters are also part of checkpoint.
    current_best_nn = AttentionChessNet().to(DEVICE)
    candidate_nn = AttentionChessNet().to(DEVICE)

    try:
        current_best_nn.load_state_dict(torch.load(net1_path, map_location=DEVICE))
        print(f"Loaded current best network from {net1_path}")
    except Exception as e:
        print(f"Error loading current_best_nn from {net1_path}: {e}. Cannot proceed with evaluation.")
        return -1.0 # Indicate failure

    try:
        candidate_nn.load_state_dict(torch.load(net2_path, map_location=DEVICE))
        print(f"Loaded candidate network from {net2_path}")
    except Exception as e:
        print(f"Error loading candidate_nn from {net2_path}: {e}. Cannot proceed with evaluation.")
        return -1.0 # Indicate failure

    current_best_nn.eval()
    candidate_nn.eval()

    chess_env = ChessEnv() # Single instance for all games

    net2_wins = 0
    draws = 0
    net1_wins = 0 # For full stats

    print(f"Starting evaluation: {num_eval_games} games between '{net1_path}' and '{net2_path}'.")
    
    for i in tqdm(range(num_eval_games), desc="Evaluation Games"):
        if i % 2 == 0:
            # Game 1: current_best_nn (net1) is White, candidate_nn (net2) is Black
            # play_eval_game returns outcome for the second net (candidate_nn)
            outcome_for_candidate, winner = play_eval_game(current_best_nn, candidate_nn, chess_env, DEVICE, num_mcts_sims_eval)
            if outcome_for_candidate == 1.0: # Candidate (Black) won
                net2_wins += 1
            elif outcome_for_candidate == -1.0: # Candidate (Black) lost (Current best (White) won)
                net1_wins +=1
            else:
                draws += 1
        else:
            # Game 2: candidate_nn (net2) is White, current_best_nn (net1) is Black
            # play_eval_game returns outcome for the second net (current_best_nn)
            outcome_for_current_best, winner = play_eval_game(candidate_nn, current_best_nn, chess_env, DEVICE, num_mcts_sims_eval)
            if outcome_for_current_best == 1.0: # Current best (Black) won
                net1_wins +=1
            elif outcome_for_current_best == -1.0: # Current best (Black) lost (Candidate (White) won)
                net2_wins += 1
            else:
                draws +=1
        
        # tqdm.write(f"Game {i+1}: Winner: {winner}, Net2 Wins: {net2_wins}, Net1 Wins: {net1_wins}, Draws: {draws}")


    total_games_played = net2_wins + net1_wins + draws
    if total_games_played == 0: return 0.0 # Avoid division by zero if no games played

    # Win rate of net2 = (net2_wins + 0.5 * draws) / total_games_played (if draws count as half point)
    # Or simply: net2_wins / total_games_played (if only wins count)
    # AlphaZero typically uses a threshold on strict wins (e.g., >55% wins for net2).
    # Let's calculate strict win rate for net2.
    win_rate_net2 = net2_wins / total_games_played
    
    print(f"\nEvaluation Results ({total_games_played} games):")
    print(f"  Candidate Network ('{net2_path}') Wins: {net2_wins}")
    print(f"  Current Best Network ('{net1_path}') Wins: {net1_wins}")
    print(f"  Draws: {draws}")
    print(f"  Candidate Network Win Rate (strict): {win_rate_net2:.4f}")

    return win_rate_net2

# --- Testing ---
if __name__ == '__main__':
    import os
    print("Evaluation Framework Testing (Conceptual)")

    DEVICE_STR = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE_STR}")

    # Create dummy model state_dict files for testing
    dummy_model_path1 = "dummy_best_net.pth"
    dummy_model_path2 = "dummy_candidate_net.pth"

    # Save dummy models (using smaller architecture for speed)
    # Ensure these models can be loaded by AttentionChessNet() default constructor
    # or pass appropriate args if architecture varies.
    # For simplicity, assume default AttentionChessNet constructor works.
    try:
        # Create smaller models for testing to speed things up
        test_config_override = {
            'd_model': 32, 'n_heads': 2, 'num_encoder_layers': 1 
        }
        dummy_net1 = AttentionChessNet(d_model=test_config_override['d_model'], 
                                     n_heads=test_config_override['n_heads'],
                                     num_encoder_layers=test_config_override['num_encoder_layers'])
        dummy_net2 = AttentionChessNet(d_model=test_config_override['d_model'],
                                     n_heads=test_config_override['n_heads'],
                                     num_encoder_layers=test_config_override['num_encoder_layers'])

        torch.save(dummy_net1.state_dict(), dummy_model_path1)
        torch.save(dummy_net2.state_dict(), dummy_model_path2)
        print(f"Saved dummy models to {dummy_model_path1} and {dummy_model_path2}")

        # Run evaluation
        num_test_eval_games = 4 # Small number for quick test
        test_mcts_sims = config.MCTS_SIMULATIONS_EVAL // 10 if config.MCTS_SIMULATIONS_EVAL > 10 else 10
        
        print(f"\nStarting evaluation with {num_test_eval_games} games, {max(10, test_mcts_sims)} MCTS sims per move...")
        
        win_rate = evaluate_networks(
            net1_path=dummy_model_path1,
            net2_path=dummy_model_path2,
            num_eval_games=num_test_eval_games,
            num_mcts_sims_eval=max(10, test_mcts_sims), # Reduced for faster test
            device_str=DEVICE_STR
        )

        print(f"\nCandidate network win rate from test: {win_rate:.4f}")

    except Exception as e:
        print(f"An error occurred during evaluation testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files
        if os.path.exists(dummy_model_path1):
            os.remove(dummy_model_path1)
        if os.path.exists(dummy_model_path2):
            os.remove(dummy_model_path2)
        print("Cleaned up dummy model files.")

    print("\nEvaluation framework conceptual test finished.")
    print("NOTE: This test relies on functional MCTS, Model, and ChessEnv. Performance of dummy models will be random.")