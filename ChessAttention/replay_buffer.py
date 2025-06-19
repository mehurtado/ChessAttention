# replay_buffer.py
# Phase 3: Self-Play Orchestration
# Component: Replay Buffer
# Objective: Store game data for training.

import torch
import random
from collections import deque
from typing import List, Tuple, NamedTuple, Optional
import numpy as np

import config

# Define the structure of a training example
class TrainingExample(NamedTuple):
    board_state_tensor: torch.Tensor  # (C, H, W) e.g. (19, 8, 8)
    mcts_policy_target: torch.Tensor  # (PolicySize,) e.g. (4672,)
    game_outcome_for_player: float    # Scalar: 1.0, -1.0, or 0.0

class ReplayBuffer:
    def __init__(self, buffer_size: int = config.REPLAY_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_game_history(self, game_history: List[Tuple[torch.Tensor, np.ndarray, float]]):
        """
        Adds a list of (board_state_tensor, mcts_policy_target_numpy, game_outcome_for_player)
        tuples from a completed game to the buffer.
        Converts mcts_policy_target from numpy to tensor.
        """
        for state_tensor, policy_target_np, outcome in game_history:
            # Ensure policy_target is a tensor
            policy_target_tensor = torch.from_numpy(policy_target_np).float()
            example = TrainingExample(state_tensor, policy_target_tensor, outcome)
            self.buffer.append(example)

    def sample_batch(self, batch_size: int = config.BATCH_SIZE) -> Optional[List[TrainingExample]]:
        """
        Samples a batch of training examples from the buffer.
        Returns None if the buffer doesn't have enough examples.
        """
        if len(self.buffer) < batch_size:
            return None
        
        return random.sample(list(self.buffer), batch_size) # Convert deque to list for random.sample

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, filepath: str):
        """Saves the replay buffer to a file."""
        # Consider saving as a list of tuples or using torch.save on the deque
        # For simplicity, let's use torch.save on the list representation
        # This might be memory intensive for very large buffers during saving/loading
        torch.save(list(self.buffer), filepath)
        print(f"Replay buffer saved to {filepath}")

    def load(self, filepath: str):
        """Loads the replay buffer from a file."""
        try:
            loaded_data = torch.load(filepath, weights_only=False)  # Use weights_only=True for efficiency
            self.buffer = deque(loaded_data, maxlen=self.buffer_size)
            print(f"Replay buffer loaded from {filepath}. Size: {len(self.buffer)}")
        except FileNotFoundError:
            print(f"Replay buffer file not found: {filepath}. Starting with an empty buffer.")
        except Exception as e:
            print(f"Error loading replay buffer: {e}. Starting with an empty buffer.")


# --- Testing ---
if __name__ == '__main__':
    import numpy as np

    print("Replay Buffer Testing")
    buffer = ReplayBuffer(buffer_size=100)

    # Create some dummy game data
    dummy_game_histories = []
    for _ in range(5): # 5 games
        game_history = []
        num_moves_in_game = random.randint(10, 20)
        final_outcome = random.choice([-1.0, 0.0, 1.0])
        for __ in range(num_moves_in_game):
            dummy_state = torch.randn(config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE)
            dummy_policy_target_np = np.random.rand(config.POLICY_OUTPUT_SIZE).astype(np.float32)
            dummy_policy_target_np /= np.sum(dummy_policy_target_np) # Normalize
            
            # Outcome is the same for all states in this simple test game history
            game_history.append((dummy_state, dummy_policy_target_np, final_outcome))
        dummy_game_histories.append(game_history)

    # Add data to buffer
    for history in dummy_game_histories:
        buffer.add_game_history(history)
    
    print(f"Buffer size after adding games: {len(buffer)}")
    assert len(buffer) <= buffer.buffer_size

    # Sample a batch
    batch_size = 32
    print(f"\nAttempting to sample a batch of size {batch_size}...")
    batch = buffer.sample_batch(batch_size)

    if batch:
        print(f"Successfully sampled a batch of {len(batch)} items.")
        first_item = batch[0]
        print(f"First item - state tensor shape: {first_item.board_state_tensor.shape}")
        print(f"First item - policy target shape: {first_item.mcts_policy_target.shape}")
        print(f"First item - game outcome: {first_item.game_outcome_for_player}")
        assert first_item.board_state_tensor.shape == (config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE)
        assert first_item.mcts_policy_target.shape == (config.POLICY_OUTPUT_SIZE,)
        assert isinstance(first_item.game_outcome_for_player, float)
    else:
        print(f"Could not sample batch. Buffer size {len(buffer)} < batch size {batch_size}.")

    # Test with buffer smaller than batch size
    small_buffer = ReplayBuffer(buffer_size=10)
    small_buffer.add_game_history(dummy_game_histories[0][:5]) # Add 5 items
    print(f"\nSmall buffer size: {len(small_buffer)}")
    small_batch = small_buffer.sample_batch(batch_size=10)
    if small_batch:
        print(f"Sampled batch of size {len(small_batch)} from small buffer (this shouldn't happen if batch_size > len).")
    else:
        print(f"Correctly did not sample batch from small buffer (size {len(small_buffer)}) when batch_size is 10 (or more).")
    
    # Test save and load
    buffer_filepath = "dummy_replay_buffer.pth"
    buffer.save(buffer_filepath)
    
    loaded_buffer = ReplayBuffer(buffer_size=100)
    loaded_buffer.load(buffer_filepath)
    
    print(f"Original buffer size: {len(buffer)}, Loaded buffer size: {len(loaded_buffer)}")
    assert len(buffer) == len(loaded_buffer)
    if len(buffer) > 0 and len(loaded_buffer) > 0:
         # Rudimentary check: compare one element if possible
         # This requires proper __eq__ on TrainingExample or comparing fields
         original_ex = list(buffer.buffer)[0]
         loaded_ex = list(loaded_buffer.buffer)[0]
         assert torch.equal(original_ex.board_state_tensor, loaded_ex.board_state_tensor)
         assert torch.equal(original_ex.mcts_policy_target, loaded_ex.mcts_policy_target)
         assert original_ex.game_outcome_for_player == loaded_ex.game_outcome_for_player
         print("First element of saved and loaded buffer matches.")

    # Clean up dummy file
    import os
    if os.path.exists(buffer_filepath):
        os.remove(buffer_filepath)
        print(f"Cleaned up {buffer_filepath}")

    print("\nReplay Buffer tests completed.")