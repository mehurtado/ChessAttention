# main.py
# Phase 6: Main Orchestration Loop (AlphaZero Cycle)
# Objective: Combine all modules into the self-improving loop.

import torch
import torch.optim as optim
import os
import time
import random
import numpy as np
from datetime import datetime

import config
from chess_env import ChessEnv
from model import AttentionChessNet
from mcts import MCTS
from replay_buffer import ReplayBuffer
from self_play import play_game
from train import training_loop # Renamed from train_model to avoid conflict
from evaluate import evaluate_networks

# For logging (TensorBoard/WandB) - conceptual
# from torch.utils.tensorboard import SummaryWriter
# import wandb

def setup_seed(seed=config.RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_alpha_zero_loop():
    setup_seed()
    DEVICE = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Starting AlphaZero Chess Engine on device: {DEVICE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Initialization ---
    print("\n--- Initializing Components ---")
    
    # Model: Start with random weights or load checkpoint
    current_best_nn = AttentionChessNet(
        input_channels=config.INPUT_CHANNELS,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        # ... other config params if they vary
    ).to(DEVICE)
    
    # Checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    initial_model_path = os.path.join(config.CHECKPOINT_DIR, "initial_model.pth")
    current_best_model_path = os.path.join(config.CHECKPOINT_DIR, "current_best_nn.pth")
    candidate_model_path = os.path.join(config.CHECKPOINT_DIR, "candidate_nn.pth")
    replay_buffer_path = os.path.join(config.CHECKPOINT_DIR, "replay_buffer.pth")

    # Attempt to load existing best model and replay buffer
    iteration_start = 0
    if os.path.exists(current_best_model_path):
        print(f"Loading existing best model from: {current_best_model_path}")
        current_best_nn.load_state_dict(torch.load(current_best_model_path, map_location=DEVICE))
        # Potentially load iteration number if saved
    else:
        print("No existing best model found. Starting with a randomly initialized model.")
        torch.save(current_best_nn.state_dict(), initial_model_path)
        torch.save(current_best_nn.state_dict(), current_best_model_path) # Save initial as best
        print(f"Saved initial random model to {initial_model_path} and {current_best_model_path}")

    # Replay Buffer
    replay_buffer = ReplayBuffer(buffer_size=config.REPLAY_BUFFER_SIZE)
    if os.path.exists(replay_buffer_path):
        print(f"Loading replay buffer from: {replay_buffer_path}")
        replay_buffer.load(replay_buffer_path)
    else:
        print("No existing replay buffer found. Starting with an empty one.")

    # Optimizer for the candidate network
    # The optimizer state should ideally be saved/loaded if resuming training of a candidate
    # For simplicity, we re-initialize it each AZ iteration for the new candidate.
    
    # Shared Chess Environment and MCTS (can be re-instantiated or passed around)
    chess_env = ChessEnv()
    # MCTS instance for self-play will use current_best_nn
    # MCTS instances for evaluation will be created with respective nets

    # Logging (conceptual)
    # timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, f"{config.PROJECT_NAME}_{timestamp}")
    # tb_writer = SummaryWriter(log_dir)
    # print(f"TensorBoard logs will be saved to: {log_dir}")
    # if using wandb:
    #   wandb.init(project=config.PROJECT_NAME, config=vars(config))


    print("\n--- Starting AlphaZero Main Loop ---")
    for az_iteration in range(iteration_start, config.TOTAL_CA_ITERATIONS):
        iteration_time_start = time.time()
        print(f"\n===== AlphaZero Iteration: {az_iteration + 1}/{config.TOTAL_CA_ITERATIONS} =====")

        # --- a. Self-Play Phase ---
        print(f"\n--- Phase: Self-Play (Iteration {az_iteration + 1}) ---")
        self_play_time_start = time.time()
        current_best_nn.eval() # Ensure model is in eval mode for MCTS inference
        mcts_self_play = MCTS(chess_env, current_best_nn, DEVICE) # MCTS with current best net

        num_new_games = config.NUM_SELF_PLAY_GAMES_PER_ITERATION
        games_generated_in_iter = 0
        for game_num in range(num_new_games):
            game_start_time = time.time()
            print(f"  Generating self-play game {game_num + 1}/{num_new_games}...")
            
            game_data = play_game(
                best_nn=current_best_nn, # Pass the nn model itself
                chess_env_instance=chess_env,
                mcts_instance=mcts_self_play,
                num_mcts_simulations=config.MCTS_SIMULATIONS_SELF_PLAY,
                C_puct=config.C_PUCT,
                dirichlet_alpha=config.DIRICHLET_ALPHA,
                dirichlet_epsilon=config.DIRICHLET_EPSILON,
                temperature_moves=config.TEMPERATURE_MOVES
            )
            if game_data:
                replay_buffer.add_game_history(game_data)
                games_generated_in_iter +=1
                print(f"  Game {game_num + 1} finished in {time.time() - game_start_time:.2f}s. Generated {len(game_data)} examples. Replay buffer size: {len(replay_buffer)}")
            else:
                print(f"  Game {game_num + 1} failed to generate data.")
            
            # Optional: Save replay buffer periodically during self-play
            if (game_num + 1) % (num_new_games // 2 + 1) == 0 : # e.g. halfway
                 replay_buffer.save(replay_buffer_path)


        print(f"Self-Play phase finished in {time.time() - self_play_time_start:.2f}s. Generated {games_generated_in_iter} new games.")
        replay_buffer.save(replay_buffer_path) # Save after all games in iter are done

        if len(replay_buffer) < config.BATCH_SIZE:
            print(f"Replay buffer size ({len(replay_buffer)}) is less than batch size ({config.BATCH_SIZE}). Skipping training and evaluation for this iteration.")
            # tb_writer.add_scalar('Iteration/SelfPlayGames', games_generated_in_iter, az_iteration)
            # tb_writer.add_scalar('ReplayBuffer/Size', len(replay_buffer), az_iteration)
            continue


        # --- b. Training Phase ---
        print(f"\n--- Phase: Training (Iteration {az_iteration + 1}) ---")
        training_time_start = time.time()
        
        # Create a candidate network (copy of current best, then train it)
        candidate_nn = AttentionChessNet(
            input_channels=config.INPUT_CHANNELS, d_model=config.D_MODEL, # etc.
        ).to(DEVICE)
        candidate_nn.load_state_dict(current_best_nn.state_dict()) # Start from current best
        
        # Optimizer for the candidate network
        optimizer = optim.AdamW(candidate_nn.parameters(), lr=config.LEARNING_RATE, weight_decay=config.L2_REG_CONST if config.L2_REG_CONST > 0 else 0)
        # Learning rate scheduler (optional but recommended)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_TRAINING_EPOCHS_PER_ITERATION * (len(replay_buffer) // config.BATCH_SIZE))

        # Determine number of training steps for this iteration
        # E.g., 1 epoch over a fraction of the buffer, or fixed number of steps
        num_steps_this_iteration = config.NUM_TRAINING_EPOCHS_PER_ITERATION * (len(replay_buffer) // config.BATCH_SIZE)
        if num_steps_this_iteration == 0 and len(replay_buffer) >= config.BATCH_SIZE:
            num_steps_this_iteration = 1 # Train at least one batch if buffer is full enough

        print(f"Training candidate network for {num_steps_this_iteration} steps...")
        training_loop( # Use the imported training_loop function
            model=candidate_nn,
            optimizer=optimizer,
            # scheduler=scheduler, # Pass if using
            replay_buffer=replay_buffer,
            num_training_steps=num_steps_this_iteration,
            batch_size=config.BATCH_SIZE,
            l2_reg_const=config.L2_REG_CONST, # Will be used if optimizer is not AdamW or AdamW wd=0
            device=DEVICE,
            # logger=tb_writer # Pass logger
        )
        torch.save(candidate_nn.state_dict(), candidate_model_path)
        print(f"Training phase finished in {time.time() - training_time_start:.2f}s. Candidate model saved to {candidate_model_path}")

        # --- c. Evaluation Phase ---
        print(f"\n--- Phase: Evaluation (Iteration {az_iteration + 1}) ---")
        eval_time_start = time.time()
        
        win_rate_candidate = evaluate_networks(
            net1_path=current_best_model_path, # Current best
            net2_path=candidate_model_path,    # Candidate
            num_eval_games=config.NUM_EVAL_GAMES,
            num_mcts_sims_eval=config.MCTS_SIMULATIONS_EVAL,
            device_str=config.DEVICE
        )
        print(f"Evaluation phase finished in {time.time() - eval_time_start:.2f}s.")
        print(f"Candidate network win rate against current best: {win_rate_candidate:.4f}")
        # tb_writer.add_scalar('Evaluation/CandidateWinRate', win_rate_candidate, az_iteration)

        if win_rate_candidate > config.EVAL_WIN_RATE_THRESHOLD:
            print(f"Candidate network is NEW BEST (Win rate: {win_rate_candidate:.4f} > {config.EVAL_WIN_RATE_THRESHOLD}). Promoting.")
            torch.save(candidate_nn.state_dict(), current_best_model_path)
            current_best_nn.load_state_dict(candidate_nn.state_dict()) # Update in-memory current_best_nn
            # tb_writer.add_text('Events/NewBestModel', f"Iteration {az_iteration+1}", az_iteration)
        else:
            print(f"Candidate network DID NOT surpass current best (Win rate: {win_rate_candidate:.4f}). Keeping current best.")

        # tb_writer.add_scalar('Iteration/SelfPlayGames', games_generated_in_iter, az_iteration)
        # tb_writer.add_scalar('ReplayBuffer/Size', len(replay_buffer), az_iteration)
        # tb_writer.add_scalar('Timing/IterationSeconds', time.time() - iteration_time_start, az_iteration)
        print(f"===== AlphaZero Iteration {az_iteration + 1} completed in {time.time() - iteration_time_start:.2f}s =====")

    # --- End of Loop ---
    print("\n--- AlphaZero Main Loop Finished ---")
    # tb_writer.close()
    # if using wandb:
    #   wandb.finish()

if __name__ == '__main__':
    try:
        main_alpha_zero_loop()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Main script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")