import torch
import random
import math
import numpy as np
from pathlib import Path

from environment import Board
from replay_memory import ReplayMemory, Transition 
from model import Model

def select_action(state, epsilon, main_network, device):
    # Choose (valid) action based on epsilon-greedy strategy 
    valid_actions = state.view(-1).nonzero().squeeze(1)
    if random.random() < epsilon:
        action = random.choice(valid_actions).item()
    else:
        with torch.no_grad():
            state = state.to(device)
            q_values = main_network(state).view(-1)
            valid_q_values = q_values.cpu()[valid_actions]
            action_idx = valid_q_values.argmax()
            action = valid_actions[action_idx].item()
    return action

def reward_function(state, action, next_state, is_terminal, width, height, n_moves):
    if is_terminal:
        reward = 0
    else:
        reward = -1
    return torch.tensor(reward).float()

width, height = 7, 9
BATCH_SIZE = 256 
GAMMA = 0.999
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 500000
TAU = 0.200
TARGET_UPDATE_STEPS = 25000
MEMORY_SIZE = 10000
REPLAY_STEPS = 10 
LR = 1e-5
CHECKPOINT_PATH = "main_network.pt"

board = Board(width, height)
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
main_network = Model(width, height).to(device)
if Path(CHECKPOINT_PATH).is_file():
    main_network.load_state_dict(torch.load(CHECKPOINT_PATH))#, map_location=device, weights_only=True))
target_network = Model(width, height).to(device)
target_network.load_state_dict(main_network.state_dict())
main_network.train()
for param in target_network.parameters():
    param.requires_grad = False

n_steps = 0
n_episodes = 0
epsilon = EPS_START
losses = []
moves_used = []
episode_rewards = []
episode_reward = 0

optimizer = torch.optim.Adam(main_network.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

while True:
    state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)
    action = select_action(state, epsilon, main_network, device)
    board.click(action)
    next_state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)
    is_terminal = board.is_game_over()
    reward = reward_function(state, action, next_state, is_terminal, width, height, board.clicks)
    episode_reward += reward.item()
    memory.push(state, action, reward, next_state, is_terminal)
    n_steps += 1

    if is_terminal:
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * n_episodes / EPS_DECAY)
        n_episodes += 1
        moves_used.append(board.clicks)
        episode_rewards.append(episode_reward)
        episode_reward = 0
        board.reset()

        if (n_episodes + 1) % 1000 == 0:
            # Print summary statistics
            mean_loss = np.mean(losses)
            mean_moves_used = np.mean(moves_used)
            mean_episode_reward = np.mean(episode_rewards)
            print(f"Episode {n_episodes + 1}, Loss: {mean_loss:.5f}, Moves used: {mean_moves_used:.2f}, Epsilon: {epsilon:.3f}, Episode reward: {mean_episode_reward:.2f}")
            losses = []
            moves_used = []
            episode_rewards = []

    if len(memory) >= BATCH_SIZE and (n_steps + 1) % REPLAY_STEPS == 0:
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        is_terminal_batch = torch.tensor(batch.is_terminal).int().unsqueeze(1).to(device)
        Q_values = main_network(state_batch).gather(1, action_batch)

        # Maxmimize over valid actions only
        target_next_batch = target_network(next_state_batch)
        target_next_batch[next_state_batch == 0] = -float("inf")
        max_next_q_values = target_next_batch.max(dim=-1, keepdim=True).values
        max_next_q_values[max_next_q_values == -float("inf")] = 0

        target_Q_values = reward_batch + GAMMA * max_next_q_values * (1 - is_terminal_batch)
        loss = torch.nn.functional.mse_loss(Q_values, target_Q_values)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        for param in main_network.parameters():
            param.grad.data.clamp_(-10, 10)
        optimizer.step()

        # Hard target update and save main policy network checkpoint
        if (n_steps + 1) % TARGET_UPDATE_STEPS == 0:
            target_network.load_state_dict(main_network.state_dict())
            torch.save(main_network.state_dict(), CHECKPOINT_PATH)

