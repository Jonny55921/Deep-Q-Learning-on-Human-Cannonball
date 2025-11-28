# Deep-Q-Learning-on-Human-Cannonball
Deep Q Learning on the Atari game Human Cannonball. We will explore a couple of models and see how they perform on this Atari game.

Reference/Adapted Code:
[https://github.com/everestso/summer25/blob/main/c166f25_02b_dqn_pong.ipynb](url)
---

# Deep Q-Networks on Human Cannonball (ALE)

This project trains Deep Q-Network (DQN) agents on the **HumanCannonball-v5** Atari environment using the Gymnasium API.

The notebook includes:

- Baseline DQN
- Double DQN (DDQN)
- Prioritized Experience Replay (PER)
- DDQN + PER (with beta-annealing)
- Hyperparameter sweeps
- Inline gameplay video recording and display

---
## Example Gameplay BEFORE Training

<video width="630" height="300" src="https://github.com/user-attachments/assets/0659b792-9b07-40d9-9ee2-00793c3555a9" controls></video>

## Example Gameplay AFTER Training

<video width="630" height="300" src="https://github.com/user-attachments/assets/1018c78a-8d1d-43cb-a122-f7b745521fc3" controls></video>

## Project Structure
```bash
.
├── VIDEOS
├── README.md # This file
└── starter_code.ipynb # Main notebook
```
markdown
Copy code

> The notebook is formatted for Google Colab + Google Drive, but can also be run locally.

---

## Environment & Dependencies

**Core Libraries**

- Python 3.9+
- PyTorch
- Gymnasium (with Atari / ALE support)
- ale-py and AutoROM for ROM management
- stable-baselines3 (for Atari wrappers)
- tensorboard
- numpy, pandas, matplotlib
- imageio, tabulate
- IPython.display (inline video support)

**Example Installation (local)**

```bash
pip install "gymnasium[atari]" ale-py autorom.accept-rom-license
pip install torch torchvision torchaudio
pip install stable-baselines3 tensorboard imageio pandas matplotlib tabulate
Atari ROMs may require running AutoROM to download and accept the license.
```
Notebook Overview
All code is contained inside starter_code.ipynb. Main sections:

## 1. Environment Description
```bash
Overview of HumanCannonball observation and action spaces
Discussion of sparse and delayed reward structure
```
## 2. Environment Setup
```bash
Imports and Colab Drive mounting
Atari environment registration
make_env(...) wrapper for preprocessing, frame stacking, and PyTorch formatting
```
## 3. Configuration & Hyperparameters
```bash
Key constants:
GAMMA, BATCH_SIZE, REPLAY_SIZE, LEARNING_RATE
EPSILON_START, EPSILON_FINAL, EPSILON_DECAY_LAST_FRAME
SYNC_TARGET_FRAMES, REPLAY_START_SIZE
Hyperparameter dictionaries (baseline_hparams, etc.)
```
## 4. Experience & Replay Buffer
```bash
Experience dataclass
ExperienceBuffer (uniform sampling)
PrioritizedReplayBuffer (PER):
append(...)
sample(...)
update_priorities(...)
```
## 5. Agent Interaction
 ```bash
Agent class:
play_step(...) handles stepping, storing transitions, and epsilon-greedy exploration
Tracks episodic rewards
```
## 6. DQN Model
```bash
DQN(nn.Module):
Three convolutional layers
Fully connected head
Outputs Q-values for all actions
```


## 7. Training Loops

7a. Baseline DQN

    train_baseline_dqn()    
    Standard TD target    
    Logs to TensorBoard (runs/baseline)

7b. Double DQN

    train_ddqn()    
    Online network selects action, target network evaluates it    
    Uses ddqn_calc_loss(...)    
    Logs to runs/ddqn

7c. PER + DDQN

    train_ddqn_per() (combined version)    
    Uses PER with importance sampling weights and beta annealing    
    Logs to runs/per

## 8. Hyperparameter Sweeps & Analysis
```bash
run_per_alpha_sweep() evaluates PER exponent alpha
smooth_curve(...) for stable return plots
TensorBoard event parsing with event_accumulator
Runs a greedy policy using the provided net.
```
