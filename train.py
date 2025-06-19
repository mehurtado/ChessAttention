# train.py
# Phase 4: Training Pipeline
# Objective: Train the neural network on data from the replay buffer.

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast # For mixed precision
from typing import Tuple, List

import config
from model import AttentionChessNet
from replay_buffer import ReplayBuffer, TrainingExample # For type hinting and data loading

def train_step(
    model: AttentionChessNet,
    optimizer: optim.Optimizer,
    scaler: GradScaler, # For mixed precision
    batch: List[TrainingExample], # List of (states, target_policies, target_values)
    l2_reg_const: float = config.L2_REG_CONST,
    device: torch.device = torch.device(config.DEVICE)
) -> Tuple[float, float, float]:
    """
    Performs a single training step on a batch of data.
    Handles mixed precision training.
    """
    model.train() # Set model to training mode

    # Unpack batch and move to device
    # states: (B, C, H, W), target_policies: (B, PolicySize), target_values: (B,)
    states_list = [item.board_state_tensor for item in batch]
    target_policies_list = [item.mcts_policy_target for item in batch]
    target_values_list = [torch.tensor([item.game_outcome_for_player], dtype=torch.float32) for item in batch]

    states = torch.stack(states_list).to(device)
    target_policies = torch.stack(target_policies_list).to(device)
    target_values = torch.stack(target_values_list).to(device).squeeze(-1) # Ensure (B,)

    optimizer.zero_grad()

    with autocast(enabled=(device.type == 'cuda')): # autocast for mixed precision if on CUDA
        pred_policy_logits, pred_values_raw = model(states) # pred_values_raw is (B, 1)
        pred_values = pred_values_raw.squeeze(-1) # Make it (B,) to match target_values

        # Policy Loss: Cross-entropy between predicted logits and MCTS policy targets
        # Target policies are distributions (probabilities from MCTS visit counts).
        # CrossEntropyLoss expects raw logits as input and class indices or probabilities as target.
        # If target_policies are probabilities, use KLDivLoss or custom CE.
    # ChessAttention uses: sum_a pi(a) * log p(a) which is CE.
        # PyTorch CrossEntropyLoss(logits, probabilities) is not standard.
        # Usually it's CrossEntropyLoss(logits, class_indices).
        # For targets as distributions: -sum(target_probs * log_softmax(logits))
        
        # Policy loss: -sum(target_probs * log_softmax(logits)) over actions, then mean over batch
        policy_loss = -torch.sum(target_policies * F.log_softmax(pred_policy_logits, dim=1), dim=1).mean()
        
        # Value Loss: Mean Squared Error between predicted values and game outcomes
        value_loss = F.mse_loss(pred_values, target_values)

        # L2 Regularization (if not handled by optimizer like AdamW)
        # AdamW's weight_decay is often preferred over manual L2 loss term.
        # If using AdamW with weight_decay > 0, this manual L2 might be redundant or even harmful.
        # If AdamW weight_decay is 0, or using SGD, then manual L2 is needed.
        l2_loss = torch.tensor(0.0).to(device)
        if l2_reg_const > 0 and not isinstance(optimizer, optim.AdamW): # Check if AdamW is not already doing it
            for param in model.parameters():
                # Exclude biases and LayerNorm/BatchNorm weights if desired
                if param.dim() > 1: # Typically only apply L2 to weights, not biases/layernorm
                    l2_loss += torch.norm(param)**2
            l2_loss *= l2_reg_const
        
        total_loss = policy_loss + value_loss + l2_loss
    
    # Backward pass and optimizer step with GradScaler
    if device.type == 'cuda':
        scaler.scale(total_loss).backward()
        if config.GRAD_CLIP_NORM is not None:
             scaler.unscale_(optimizer) # Unscale gradients before clipping
             torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
    else: # CPU training
        total_loss.backward()
        if config.GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()

    return policy_loss.item(), value_loss.item(), total_loss.item()

def training_loop(
    model: AttentionChessNet,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    num_training_steps: int, # Or epochs
    batch_size: int = config.BATCH_SIZE,
    l2_reg_const: float = config.L2_REG_CONST,
    device: torch.device = torch.device(config.DEVICE),
    # logger=None # For TensorBoard/WandB
):
    """
    Main training loop for a certain number of steps or epochs.
    """
    model.to(device)
    scaler = GradScaler(enabled=(device.type == 'cuda')) # Initialize GradScaler

    for step in range(num_training_steps):
        if len(replay_buffer) < batch_size:
            # print(f"Step {step+1}/{num_training_steps}: Replay buffer too small ({len(replay_buffer)}/{batch_size}). Skipping training step.")
            # Consider waiting or reducing batch size if this happens often early on.
            # For now, just skip.
            if step == 0 : print(f"Replay buffer too small ({len(replay_buffer)}/{batch_size}). Waiting for more data...")
            torch.cuda.empty_cache() # Free up memory if possible
            continue # Or break, or wait with time.sleep(10)

        batch = replay_buffer.sample_batch(batch_size)
        if batch is None: # Should be caught by len check, but for safety
            # print(f"Step {step+1}/{num_training_steps}: Failed to sample batch. Skipping.")
            continue
            
        policy_loss, value_loss, total_loss = train_step(
            model, optimizer, scaler, batch, l2_reg_const, device
        )

        if (step + 1) % 50 == 0: # Log every 50 steps
            print(f"Training Step: {step+1}/{num_training_steps}, "
                  f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Total Loss: {total_loss:.4f}")
            # if logger:
            #     logger.add_scalar('Loss/Policy', policy_loss, step)
            #     logger.add_scalar('Loss/Value', value_loss, step)
            #     logger.add_scalar('Loss/Total', total_loss, step)
            #     logger.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], step)
    
    print("Training loop finished.")


# --- Testing ---
if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm

    print("Training Pipeline Testing")

    DEVICE = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Initialize components
    # Use smaller model/buffer for faster testing if needed
    test_model = AttentionChessNet(
        d_model=64, num_encoder_layers=2, n_heads=2 # Smaller model for test
    ).to(DEVICE)
    
    # Optimizer: AdamW is generally good.
    # If using AdamW, set l2_reg_const to 0 in train_step if weight_decay is used here.
    test_optimizer = optim.AdamW(test_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.L2_REG_CONST if config.L2_REG_CONST > 0 else 0)
    # If L2_REG_CONST is 0, AdamW's weight_decay will be 0.
    # If L2_REG_CONST > 0, AdamW handles L2. The manual l2_loss in train_step should ideally be off.
    # For this test, let's assume l2_reg_const in train_step is primary if optimizer is not AdamW,
    # or if AdamW has weight_decay=0.
    # For simplicity, let AdamW handle it by setting its weight_decay.
    # The train_step's l2_loss will only activate if optimizer is not AdamW.

    test_replay_buffer = ReplayBuffer(buffer_size=1000) # Smaller buffer

    # Populate buffer with some dummy data
    print("Populating dummy replay buffer...")
    num_dummy_items = config.BATCH_SIZE * 5 # Enough for a few batches
    for _ in tqdm(range(num_dummy_items)):
        dummy_state = torch.randn(config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE)
        dummy_policy_target_np = np.random.rand(config.POLICY_OUTPUT_SIZE).astype(np.float32)
        dummy_policy_target_np /= np.sum(dummy_policy_target_np)
        dummy_outcome = random.choice([-1.0, 0.0, 1.0])
        # add_game_history expects a list of game histories, each being a list of tuples
        # For simplicity, add one item as one "game" of one step
        test_replay_buffer.add_game_history([(dummy_state, dummy_policy_target_np, dummy_outcome)])

    print(f"Dummy replay buffer populated with {len(test_replay_buffer)} items.")

    # Start training loop
    num_test_steps = 100 # Short training loop for test
    print(f"\nStarting training loop for {num_test_steps} steps...")
    
    training_loop(
        model=test_model,
        optimizer=test_optimizer,
        replay_buffer=test_replay_buffer,
        num_training_steps=num_test_steps,
        batch_size=config.BATCH_SIZE // 2 if config.BATCH_SIZE > 32 else config.BATCH_SIZE, # Smaller batch for test
        l2_reg_const=config.L2_REG_CONST, # This will be used if optimizer is not AdamW, or AdamW has wd=0
        device=DEVICE
    )

    print("\nTraining pipeline conceptual test finished.")
    print("NOTE: Ensure model, optimizer, and replay buffer are correctly initialized.")
    print("If using AdamW with weight_decay, the manual L2 in train_step might be redundant.")