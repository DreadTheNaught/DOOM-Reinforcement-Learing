# DOOM-Reinforcement-Learning

## Overview
This project implements reinforcement learning (RL) using the ViZDoom environment. The goal is to train an agent to play DOOM using reinforcement learning techniques, particularly PPO (Proximal Policy Optimization) and DQN (Deep Q-Learning).

## Features
- **ViZDoom Environment Setup** for AI-based gameplay.
- **PPO Algorithm Implementation** using reinforcement learning.
- **Logging and Checkpointing** for training progress tracking.
- **Customizable Configuration** via JSON files.
- **Callback Support** for training monitoring.

## Repository Structure
```
├── checkpoint/             # Saved models and training checkpoints
├── logs/PPO_1/             # Training logs and statistics
├── utils/                  # Utility scripts
├── .gitignore              # Ignored files and directories
├── _vizdoom.ini            # Configuration file for ViZDoom environment
├── callback.py             # Callback functions for monitoring training
├── config.json             # Configuration settings for training
├── main.py                 # Entry point for training the agent
├── readme.md               # Project documentation
├── trainer.py              # Training script for reinforcement learning
```

## Usage
1. **Configure Settings**: Modify `config.json` for environment settings.
2. **Train the Agent**:
   ```bash
   python main.py
   ```
3. **Monitor Training**: Logs are stored in `logs/PPO_1/`.
4. **Load & Evaluate**:
   - Use `checkpoint/` to load saved models.
   - Run evaluation using `trainer.py`.

## Dependencies
- Python 3.x
- ViZDoom
- Stable-Baselines3 (for PPO implementation)
- OpenAI Gym
- NumPy
- TensorFlow/PyTorch

## Contributing
Feel free to submit pull requests or report issues.

## License
MIT License

## Acknowledgments
- Based on reinforcement learning principles and ViZDoom simulation.
- Inspired by prior work in game-playing AI.


