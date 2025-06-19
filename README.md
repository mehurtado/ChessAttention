# Project Title: ChessAttention

## Description
This project implements a chess-playing AI based on the ChessAttention algorithm, utilizing an attention-based neural network (Transformer) for policy and value prediction, and Monte Carlo Tree Search (MCTS) for move selection. The system learns and improves by playing games against itself.

## Features
- **ChessAttention Algorithm**: Implements the core self-play, training, and evaluation loop.
- **Attention-Based Neural Network**: Uses a Transformer encoder architecture for learning chess patterns, policy (move probabilities), and value (game outcome prediction).
- **Monte Carlo Tree Search (MCTS)**: Employs MCTS for guiding move selection during self-play and evaluation, balancing exploration and exploitation.
- **Self-Play Learning**: Generates training data by playing games against its current best version.
- **Replay Buffer**: Stores game positions and outcomes for training the neural network.
- **Chess Environment Wrapper**: Uses `python-chess` for chess logic and provides a custom environment for board representation and move handling.
- **Configuration Management**: Centralized configuration for hyperparameters and settings (`config.py`).
- **Training Pipeline**: Includes scripts for training the neural network on data from the replay buffer.
- **Evaluation Framework**: Provides tools to evaluate the strength of new network checkpoints against previous versions.
- **Utilities**: Includes logging and device management utilities.

## Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AttentionalAlphaZeroChess
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Or your specific CUDA/CPU version
    pip install python-chess numpy tqdm
    ```
    (Note: Adjust PyTorch installation command based on your system (CUDA version or CPU-only) by visiting https://pytorch.org/)

## Usage
The main entry point for the ChessAttention training loop is `main.py`.

```bash
python main.py
```
This script will:
1.  Initialize the neural network (either randomly or from a checkpoint).
2.  Start the ChessAttention loop:
    *   **Self-Play**: Generate games using the current best network and MCTS, storing data in the replay buffer.
    *   **Training**: Train a new candidate network on data sampled from the replay buffer.
    *   **Evaluation**: Evaluate the candidate network against the current best network. If the candidate is significantly better, it becomes the new best network.
3.  Checkpoints for the model and replay buffer will be saved in the `./checkpoints` directory (by default).
4.  (Conceptual) TensorBoard logs may be saved in `./runs`.

### Key Scripts:
-   `main.py`: Orchestrates the main ChessAttention loop.
-   `train.py`: Handles the training of the neural network.
-   `self_play.py`: Manages the self-play game generation.
-   `evaluate.py`: Compares the performance of two network versions.
-   `config.py`: Modify hyperparameters and configurations here.

## File Structure
```
AttentionalChessAttention/
├── README.md                 # This file
├── config.py                 # Centralized configuration for hyperparameters
├── main.py                   # Main script for running the ChessAttention loop
├── model.py                  # Neural network architecture (AttentionChessNet)
├── mcts.py                   # Monte Carlo Tree Search implementation
├── chess_env.py              # Chess environment wrapper using python-chess
├── self_play.py              # Logic for self-play game generation
├── replay_buffer.py          # Replay buffer for storing training data
├── train.py                  # Training loop for the neural network
├── evaluate.py               # Framework for evaluating network checkpoints
├── utils.py                  # Utility functions (e.g., logging, device selection)
├── requirements.txt          # (Optional) For listing dependencies
└── checkpoints/              # (Created at runtime) For saving model and buffer states
└── runs/                     # (Conceptual/Created at runtime) For TensorBoard logs
```

## Contributing
Contributions are welcome! Please follow these general guidelines:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes, ensuring code is well-commented and follows a consistent style.
4.  Add or update tests if applicable.
5.  Submit a pull request with a clear description of your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (if one is created, otherwise assume MIT).
