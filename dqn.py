import torch
import random
import math

from environment import Board
from replay_memory import ReplayMemory, Transition 
from model import Model

def select_action(state, epsilon, main_network):
    # Choose (valid) action based on epsilon-greedy strategy 
    valid_actions = state.view(-1).nonzero().squeeze(1)
    if random.random() < epsilon:
        return random.choice(valid_actions).item()
    with torch.no_grad():
        return valid_actions[main_network(state).view(-1)[valid_actions].argmax().item()]


width, height = 3, 4#7, 9
board = Board(width, height)
n_actions = width * height

main_network = Model(width, height)
target_network = Model(width, height)
#target_network.load_state_dict(main_network.state_dict())

main_network.train()
target_network.eval()


BATCH_SIZE = 64 
GAMMA = 0.90
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 100000 
TAU = 0.01
LR = 1e-3

n_episodes = 0
epsilon = EPS_START

losses = []
moves_used = []
episode_rewards = []
episode_reward = 0

optimizer = torch.optim.AdamW(main_network.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

while True:
    state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)
    action = select_action(state, epsilon, main_network)
    board.click(action)
    next_state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)

    is_terminal = board.is_game_over()
    n_blocks_removed = (next_state == 0).sum().item() - (state == 0).sum().item()
    reward = torch.tensor(10 if is_terminal else -2).float()
    episode_reward += reward.item()
    memory.push(state, action, reward, next_state, is_terminal)

    state = next_state

    if is_terminal:
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * n_episodes / EPS_DECAY)
        n_episodes += 1
        moves_used.append(board.clicks)
        episode_rewards.append(episode_reward)
        episode_reward = 0
        board.reset()

        if len(memory) < BATCH_SIZE:
            continue

        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1)
        next_state_batch = torch.cat(batch.next_state)
        is_terminal_batch = torch.tensor(batch.is_terminal).int().unsqueeze(1)
        Q_values = main_network(state_batch).gather(1, action_batch)
        target_Q_values = reward_batch + GAMMA * target_network(next_state_batch).max(dim=1, keepdim=True).values * (1 - is_terminal_batch)
        
        loss = torch.nn.functional.smooth_l1_loss(Q_values, target_Q_values)
        #loss = torch.nn.functional.mse_loss(Q_values, target_Q_values)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        #for param in main_network.parameters():
        #    param.grad.data.clamp_(-1, 1)
        optimizer.step()


        target_state_dict = target_network.state_dict()
        main_network_state_dict = main_network.state_dict()
        for key in main_network_state_dict:
            target_state_dict[key] = main_network_state_dict[key]*TAU + target_state_dict[key]*(1-TAU)
        target_network.load_state_dict(target_state_dict)

        if (n_episodes + 1) % 100 == 0:
            mean_loss = sum(losses) / len(losses)
            mean_moves_used = sum(moves_used) / len(moves_used)
            mean_episode_reward = sum(episode_rewards) / len(episode_rewards)
            print(f"Episode {n_episodes + 1}, Loss: {mean_loss:.4f}, Moves used: {mean_moves_used:.1f}, Epsilon: {epsilon:.3f}, Episode reward: {mean_episode_reward:.1f}")
            losses = []
            moves_used = []



"""
n_episodes = 0

START_EPISODE:

Initalize board to state S

PICK_ACTION:

Choose valid action A based on epsilon-greedy strategy:
	with probability epsilon pick a random action
	with probability 1 - epsilon pick the action with the highest Q-value from the main network

Perform action A
Get reward R = reward(S, A)
Observe new state S'
T = is S' is_terminal?

Store (S,A,R,S',T) into replay memory

Compute loss value based on reward and target network output
Optimize main network

IF T:
	n_episodes += 1
	REPLAY
	START_EPISODE
	if (n_episodes + 1) % 100 == 0:
		update target network: copy weights from main network to target network
ELSE:
	PICK_ACTION
"""
