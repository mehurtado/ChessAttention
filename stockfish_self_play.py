
#!/usr/bin/env python3
"""
Enhanced Stockfish Reward Self-Play Script for Chess AI

This script implements self-play where the chess AI plays against itself,
receiving immediate rewards for EACH move based on Stockfish evaluation at depth 10.
The rewards are normalized centipawn evaluations, providing granular feedback.

Key Features:
- Self-play with Stockfish evaluation for each move
- Normalized centipawn rewards using tanh(centipawns/400) scaling
- Both white and black players receive appropriate rewards
- Depth 10 Stockfish analysis for move evaluation
- Enhanced error handling and logging
- Integration with existing training pipeline
- Configurable Stockfish parameters
- Comprehensive statistics tracking

Requirements:
- python-chess
- stockfish (executable)
- pytorch
- numpy

Author: AI Assistant
Date: July 18, 2025
Version: 2.0 (Enhanced)
"""

import os
import sys
import argparse
import logging
import math
import time
import pickle
import json
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine
from pathlib import Path

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing chess AI modules
try:
    from chess_env import ChessEnv
    from model import AttentionChessNet
    from mcts import MCTS
    import config as config
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Please ensure the following files are in the same directory:")
    print("- chess_env_simple.py")
    print("- model_simple.py") 
    print("- mcts_simple.py")
    print("- config_simple.py")
    sys.exit(1)


class StockfishConfig:
    """Configuration class for Stockfish integration."""
    
    def __init__(self):
        # Stockfish settings
        self.stockfish_path = "/usr/games/stockfish"
        self.stockfish_depth = 10
        self.stockfish_time_limit = 1.0  # seconds per evaluation
        self.stockfish_threads = 1
        self.stockfish_hash_size = 128  # MB
        
        # Reward normalization settings
        self.centipawn_scale = 400.0  # For tanh normalization: tanh(cp/400)
        self.mate_reward = 0.95  # Reward for mate positions
        self.reward_blend_ratio = 0.7  # 0.7 Stockfish + 0.3 game outcome
        
        # Training settings
        self.max_training_data = 100000
        self.batch_size = 256
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.training_epochs = 5
        
        # Self-play settings
        self.temperature_moves = 30
        self.max_game_moves = 400
        self.mcts_simulations = 800
        
        # Logging and saving
        self.log_level = "INFO"
        self.save_interval = 5
        self.checkpoint_dir = "./checkpoints"
        self.log_file = "stockfish_selfplay.log"


class EnhancedStockfishSelfPlay:
    """
    Enhanced self-play system with Stockfish-based move evaluation.
    
    This class provides improved integration with the existing chess AI architecture,
    better error handling, and more sophisticated reward calculation.
    """
    
    def __init__(self, config_obj: StockfishConfig = None, model_path: str = None):
        """
        Initialize the enhanced Stockfish self-play system.
        
        Args:
            config_obj: Configuration object with all settings
            model_path: Path to pre-trained model (optional)
        """
        self.config = config_obj or StockfishConfig()
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize chess environment and model
        self.chess_env = ChessEnv()
        self.model = AttentionChessNet()
        self.model.to(self.device)
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            self.logger.info(f"Loaded model from {model_path}")
        else:
            self.logger.info("Using randomly initialized model")
        
        # Initialize MCTS
        self.mcts = MCTS(self.chess_env, self.model, self.device)
        
        # Initialize Stockfish
        self.engine = None
        self.init_stockfish()
        
        # Training data storage
        self.training_data = deque(maxlen=self.config.max_training_data)
        
        # Enhanced statistics tracking
        self.stats = defaultdict(int)
        self.stats.update({
            'games_played': 0,
            'total_moves': 0,
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'stockfish_evaluations': 0,
            'training_updates': 0,
            'engine_errors': 0,
            'mcts_errors': 0,
            'total_training_time': 0.0,
            'total_game_time': 0.0
        })
        
        # Reward statistics
        self.reward_stats = {
            'stockfish_rewards': [],
            'game_outcomes': [],
            'combined_rewards': []
        }
    
    def setup_logging(self):
        """Setup enhanced logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Setup file handler
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            handlers=[file_handler, console_handler]
        )
    
    def init_stockfish(self):
        """Initialize Stockfish engine with enhanced configuration."""
        try:
            # Check if Stockfish executable exists
            if not os.path.exists(self.config.stockfish_path):
                raise FileNotFoundError(f"Stockfish not found at {self.config.stockfish_path}")
            
            # Initialize engine
            self.engine = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)
            
            # Configure engine options
            self.engine.configure({
                "Threads": self.config.stockfish_threads,
                "Hash": self.config.stockfish_hash_size
            })
            
            # Test engine
            test_board = chess.Board()
            info = self.engine.analyse(test_board, chess.engine.Limit(depth=1))
            
            self.logger.info(f"Stockfish initialized successfully:")
            self.logger.info(f"  Path: {self.config.stockfish_path}")
            self.logger.info(f"  Depth: {self.config.stockfish_depth}")
            self.logger.info(f"  Threads: {self.config.stockfish_threads}")
            self.logger.info(f"  Hash: {self.config.stockfish_hash_size}MB")
            self.logger.info(f"  Test evaluation: {info.get('score', 'No score')}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Stockfish: {e}")
            self.logger.error("Please ensure Stockfish is installed and accessible")
            raise
    
    def get_stockfish_evaluation(self, board: chess.Board) -> float:
        """
        Get normalized centipawn evaluation from Stockfish.
        
        Uses tanh(centipawns/400) normalization as specified in requirements.
        
        Args:
            board: Chess board position to evaluate
            
        Returns:
            Normalized evaluation score between -1 and 1
        """
        try:
            # Analyze position with Stockfish
            limit = chess.engine.Limit(
                depth=self.config.stockfish_depth,
                time=self.config.stockfish_time_limit
            )
            info = self.engine.analyse(board, limit)
            score = info.get('score')
            
            if score is None:
                self.logger.warning("Stockfish returned no score")
                return 0.0
            
            # Get relative score (from current player's perspective)
            relative_score = score.relative
            
            if relative_score.is_mate():
                # Handle mate scores
                mate_in = relative_score.mate()
                if mate_in > 0:
                    return self.config.mate_reward  # Winning
                else:
                    return -self.config.mate_reward  # Losing
            else:
                # Handle centipawn scores
                cp_score = relative_score.score(mate_score=10000)
                
                if cp_score is None:
                    return 0.0
                
                # Normalize using tanh(cp/400) as specified
                normalized_score = math.tanh(cp_score / self.config.centipawn_scale)
                return max(-0.99, min(0.99, normalized_score))  # Clamp to avoid extreme values
                
        except Exception as e:
            self.logger.error(f"Stockfish evaluation error: {e}")
            self.stats['engine_errors'] += 1
            return 0.0
    
    def play_single_game(self) -> List[Tuple[torch.Tensor, np.ndarray, float, float]]:
        """
        Play a single self-play game with enhanced Stockfish evaluation.
        
        Returns:
            List of training examples: (state, policy, stockfish_reward, game_outcome)
        """
        game_start_time = time.time()
        self.chess_env.reset()
        game_data = []
        move_count = 0
        
        self.logger.debug("Starting new self-play game")
        
        while not self.chess_env.is_game_over() and move_count < self.config.max_game_moves:
            try:
                current_board = self.chess_env.board.copy()
                current_player = current_board.turn
                
                # Get board representation
                board_tensor = self.chess_env.board_to_input_tensor(current_player)
                
                # Run MCTS
                use_temperature = move_count < self.config.temperature_moves
                temperature = 1.0 if use_temperature else 0.1
                
                print("test")
                chosen_move, mcts_policy, _ = self.mcts.run_simulations(
                    root_board_state=current_board,
                    num_simulations=self.config.mcts_simulations,
                    C_puct=self.config.C_PUCT,
                    dirichlet_alpha=self.config.DIRICHLET_ALPHA,
                    dirichlet_epsilon=self.config.DIRICHLET_EPSILON,
                )
                print("test2")
                
                if chosen_move is None:
                    self.logger.warning(f"MCTS returned no move at move {move_count}")
                    break
                
                # Make the move
                self.chess_env.make_move(chosen_move)
                
                # Get Stockfish evaluation of resulting position
                stockfish_reward = self.get_stockfish_evaluation(self.chess_env.board)
                
                # Adjust reward for player perspective
                # White gets positive reward for positive evaluation
                # Black gets positive reward for negative evaluation
                if current_player == chess.BLACK:
                    stockfish_reward = -stockfish_reward
                
                # Store training example (game outcome will be filled later)
                game_data.append((
                    board_tensor.clone(),
                    mcts_policy.copy(),
                    stockfish_reward,
                    0.0  # Placeholder for game outcome
                ))
                
                move_count += 1
                self.stats['total_moves'] += 1
                self.stats['stockfish_evaluations'] += 1
                
                # Log progress periodically
                if move_count % 20 == 0:
                    self.logger.debug(
                        f"Move {move_count}: {chosen_move}, "
                        f"Stockfish reward: {stockfish_reward:.3f}"
                    )
                
            except Exception as e:
                self.logger.error(f"Error during move {move_count}: {e}")
                self.stats['mcts_errors'] += 1
                break
        
        # Determine final game outcome
        game_outcome = self.chess_env.get_game_outcome()
        if game_outcome is None:
            game_outcome = 0.0  # Draw or incomplete game
        
        # Update game statistics
        self.stats['games_played'] += 1
        if game_outcome > 0:
            self.stats['white_wins'] += 1
        elif game_outcome < 0:
            self.stats['black_wins'] += 1
        else:
            self.stats['draws'] += 1
        
        # Assign final game outcomes and create combined rewards
        final_game_data = []
        for i, (state, policy, stockfish_reward, _) in enumerate(game_data):
            # Determine player for this move
            move_player = chess.WHITE if i % 2 == 0 else chess.BLACK
            
            # Game outcome from player's perspective
            if move_player == chess.WHITE:
                player_outcome = game_outcome
            else:
                player_outcome = -game_outcome
            
            # Combine Stockfish reward with game outcome
            combined_reward = (
                self.config.reward_blend_ratio * stockfish_reward +
                (1 - self.config.reward_blend_ratio) * player_outcome
            )
            
            final_game_data.append((state, policy, stockfish_reward, combined_reward))
            
            # Update reward statistics
            self.reward_stats['stockfish_rewards'].append(stockfish_reward)
            self.reward_stats['game_outcomes'].append(player_outcome)
            self.reward_stats['combined_rewards'].append(combined_reward)
        
        game_time = time.time() - game_start_time
        self.stats['total_game_time'] += game_time
        
        self.logger.info(
            f"Game completed: {move_count} moves, "
            f"outcome: {game_outcome}, time: {game_time:.2f}s"
        )
        
        return final_game_data
    
    def run_self_play_episodes(self, num_episodes: int) -> None:
        """
        Run multiple self-play episodes with enhanced tracking.
        
        Args:
            num_episodes: Number of games to play
        """
        self.logger.info(f"Starting {num_episodes} self-play episodes")
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            try:
                # Play a single game
                game_data = self.play_single_game()
                
                # Add to training data
                self.training_data.extend(game_data)
                
                episode_time = time.time() - episode_start_time
                
                self.logger.info(
                    f"Episode {episode + 1}/{num_episodes} completed in {episode_time:.2f}s, "
                    f"collected {len(game_data)} training examples"
                )
                
                # Periodic operations
                if (episode + 1) % self.config.save_interval == 0:
                    self.save_training_data(f"training_data_episode_{episode + 1}.pkl")
                    self.train_model()
                    self.save_checkpoint(f"checkpoint_episode_{episode + 1}.pth")
                    self.print_statistics()
                    self.save_statistics(f"stats_episode_{episode + 1}.json")
                
            except Exception as e:
                self.logger.error(f"Error in episode {episode + 1}: {e}")
                continue
        
        self.logger.info("Self-play episodes completed")
    
    def train_model(self) -> None:
        """
        Train the neural network with enhanced loss calculation.
        """
        if len(self.training_data) < self.config.batch_size:
            self.logger.warning(
                f"Insufficient training data: {len(self.training_data)} < {self.config.batch_size}"
            )
            return
        
        training_start_time = time.time()
        self.logger.info(f"Training model on {len(self.training_data)} examples")
        
        # Prepare training data
        states = []
        policies = []
        combined_rewards = []
        
        for state, policy, _, combined_reward in self.training_data:
            states.append(state)
            policies.append(policy)
            combined_rewards.append(combined_reward)
        
        states = torch.stack(states).to(self.device)
        policies = torch.tensor(np.array(policies), dtype=torch.float32).to(self.device)
        values = torch.tensor(combined_rewards, dtype=torch.float32).to(self.device)
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        self.model.train()
        dataset_size = len(states)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0
        
        for epoch in range(self.config.training_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_batches = 0
            
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                
                batch_states = states[batch_indices]
                batch_policies = policies[batch_indices]
                batch_values = values[batch_indices]
                
                # Forward pass
                pred_policies, pred_values = self.model(batch_states)
                
                # Calculate losses
                policy_loss = nn.CrossEntropyLoss()(pred_policies, batch_policies)
                value_loss = nn.MSELoss()(pred_values, batch_values)
                total_loss = policy_loss + value_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_batches += 1
            
            avg_policy_loss = epoch_policy_loss / epoch_batches
            avg_value_loss = epoch_value_loss / epoch_batches
            
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.training_epochs}: "
                f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}"
            )
            
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_batches += epoch_batches
        
        training_time = time.time() - training_start_time
        self.stats['training_updates'] += 1
        self.stats['total_training_time'] += training_time
        
        self.logger.info(
            f"Training completed in {training_time:.2f}s. "
            f"Avg Policy Loss: {total_policy_loss/total_batches:.4f}, "
            f"Avg Value Loss: {total_value_loss/total_batches:.4f}"
        )
    
    def save_training_data(self, filename: str) -> None:
        """Save training data with error handling."""
        try:
            filepath = os.path.join(self.config.checkpoint_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(list(self.training_data), f)
            self.logger.info(f"Training data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save training data: {e}")
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint with comprehensive data."""
        try:
            filepath = os.path.join(self.config.checkpoint_dir, filename)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'stats': dict(self.stats),
                'reward_stats': self.reward_stats,
                'config': self.config.__dict__,
                'training_data_size': len(self.training_data)
            }, filepath)
            self.logger.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_model(self, filename: str) -> None:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'stats' in checkpoint:
                self.stats.update(checkpoint['stats'])
            if 'reward_stats' in checkpoint:
                self.reward_stats.update(checkpoint['reward_stats'])
                
            self.logger.info(f"Model loaded from {filename}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def save_statistics(self, filename: str) -> None:
        """Save detailed statistics to JSON file."""
        try:
            filepath = os.path.join(self.config.checkpoint_dir, filename)
            
            # Calculate additional statistics
            stats_data = dict(self.stats)
            
            if self.reward_stats['stockfish_rewards']:
                stats_data['reward_statistics'] = {
                    'avg_stockfish_reward': np.mean(self.reward_stats['stockfish_rewards']),
                    'std_stockfish_reward': np.std(self.reward_stats['stockfish_rewards']),
                    'avg_combined_reward': np.mean(self.reward_stats['combined_rewards']),
                    'std_combined_reward': np.std(self.reward_stats['combined_rewards'])
                }
            
            if stats_data['games_played'] > 0:
                stats_data['win_rates'] = {
                    'white_win_rate': stats_data['white_wins'] / stats_data['games_played'],
                    'black_win_rate': stats_data['black_wins'] / stats_data['games_played'],
                    'draw_rate': stats_data['draws'] / stats_data['games_played']
                }
                stats_data['avg_game_length'] = stats_data['total_moves'] / stats_data['games_played']
            
            with open(filepath, 'w') as f:
                json.dump(stats_data, f, indent=2)
                
            self.logger.info(f"Statistics saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")
    
    def print_statistics(self) -> None:
        """Print comprehensive statistics."""
        print("\n" + "="*80)
        print("ENHANCED STOCKFISH REWARD SELF-PLAY STATISTICS")
        print("="*80)
        
        # Game statistics
        print(f"Games Played: {self.stats['games_played']}")
        print(f"Total Moves: {self.stats['total_moves']}")
        print(f"White Wins: {self.stats['white_wins']}")
        print(f"Black Wins: {self.stats['black_wins']}")
        print(f"Draws: {self.stats['draws']}")
        
        if self.stats['games_played'] > 0:
            avg_game_length = self.stats['total_moves'] / self.stats['games_played']
            white_wr = self.stats['white_wins'] / self.stats['games_played']
            black_wr = self.stats['black_wins'] / self.stats['games_played']
            draw_rate = self.stats['draws'] / self.stats['games_played']
            
            print(f"Average Game Length: {avg_game_length:.1f}")
            print(f"White Win Rate: {white_wr:.3f}")
            print(f"Black Win Rate: {black_wr:.3f}")
            print(f"Draw Rate: {draw_rate:.3f}")
        
        # Training statistics
        print(f"\nTraining Updates: {self.stats['training_updates']}")
        print(f"Training Data Size: {len(self.training_data)}")
        print(f"Total Training Time: {self.stats['total_training_time']:.2f}s")
        print(f"Total Game Time: {self.stats['total_game_time']:.2f}s")
        
        # Engine statistics
        print(f"\nStockfish Evaluations: {self.stats['stockfish_evaluations']}")
        print(f"Engine Errors: {self.stats['engine_errors']}")
        print(f"MCTS Errors: {self.stats['mcts_errors']}")
        
        # Reward statistics
        if self.reward_stats['stockfish_rewards']:
            sf_rewards = self.reward_stats['stockfish_rewards']
            combined_rewards = self.reward_stats['combined_rewards']
            
            print(f"\nReward Statistics:")
            print(f"Avg Stockfish Reward: {np.mean(sf_rewards):.4f} ± {np.std(sf_rewards):.4f}")
            print(f"Avg Combined Reward: {np.mean(combined_rewards):.4f} ± {np.std(combined_rewards):.4f}")
            print(f"Stockfish Reward Range: [{np.min(sf_rewards):.4f}, {np.max(sf_rewards):.4f}]")
        
        print("="*80 + "\n")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.engine:
            try:
                self.engine.quit()
                self.logger.info("Stockfish engine closed")
            except Exception as e:
                self.logger.error(f"Error closing Stockfish engine: {e}")


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced Stockfish Reward Self-Play for Chess AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of self-play episodes")
    parser.add_argument("--mcts-sims", type=int, default=800,
                       help="MCTS simulations per move")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to pre-trained model")
    
    # Stockfish parameters
    parser.add_argument("--stockfish-path", type=str, default="/usr/games/stockfish",
                       help="Path to Stockfish executable")
    parser.add_argument("--stockfish-depth", type=int, default=10,
                       help="Stockfish analysis depth")
    parser.add_argument("--centipawn-scale", type=float, default=400.0,
                       help="Centipawn scale for tanh normalization")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--training-epochs", type=int, default=5,
                       help="Training epochs per update")
    
    # Other parameters
    parser.add_argument("--save-interval", type=int, default=5,
                       help="Save interval (episodes)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Create configuration
    config_obj = StockfishConfig()
    config_obj.stockfish_path = args.stockfish_path
    config_obj.stockfish_depth = args.stockfish_depth
    config_obj.centipawn_scale = args.centipawn_scale
    config_obj.batch_size = args.batch_size
    config_obj.learning_rate = args.learning_rate
    config_obj.training_epochs = args.training_epochs
    config_obj.save_interval = args.save_interval
    config_obj.log_level = args.log_level
    config_obj.checkpoint_dir = args.checkpoint_dir
    config_obj.mcts_simulations = args.mcts_sims
    
    print("Enhanced Stockfish Reward Self-Play Training")
    print("="*60)
    print(f"Episodes: {args.episodes}")
    print(f"MCTS Simulations: {args.mcts_sims}")
    print(f"Stockfish Depth: {args.stockfish_depth}")
    print(f"Centipawn Scale: {args.centipawn_scale}")
    print(f"Model Path: {args.model_path or 'None (random init)'}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print("="*60)
    
    # Initialize and run self-play
    selfplay = None
    try:
        selfplay = EnhancedStockfishSelfPlay(config_obj, args.model_path)
        
        # Run self-play episodes
        selfplay.run_self_play_episodes(args.episodes)
        
        # Final operations
        selfplay.print_statistics()
        selfplay.save_checkpoint("final_model_stockfish_reward.pth")
        selfplay.save_training_data("final_training_data_stockfish_reward.pkl")
        selfplay.save_statistics("final_statistics.json")
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if selfplay:
            selfplay.cleanup()


if __name__ == "__main__":
    main()
