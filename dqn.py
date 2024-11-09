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


def reward_function(state, action, next_state, is_terminal):
    if is_terminal:
        reward = 1
    else:
        reward = -0.1

    return torch.tensor(reward).float()

width, height = 3, 4#7, 9
board = Board(width, height)
n_actions = width * height

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

main_network = Model(width, height).to(device)
target_network = Model(width, height).to(device)
target_network.load_state_dict(main_network.state_dict())


BATCH_SIZE = 128#64 
GAMMA = 0.99
EPS_START = 0.999
EPS_END = 0.05
EPS_DECAY = 100000#10000 
TAU = 0.001
LR = 1e-5

n_episodes = 0
epsilon = EPS_START

losses = []
moves_used = []
episode_rewards = []
episode_reward = 0

optimizer = torch.optim.Adam(main_network.parameters(), lr=LR)
memory = ReplayMemory(5000)

while True:
    state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)
    action = select_action(state, epsilon, main_network)
    board.click(action)
    next_state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)
    is_terminal = board.is_game_over()
    
    #print(f"Original state:\n{state.view(height, width)}")
    #print(f"Action: {action} (x={action % width}, y={action // width})")
    #print(f"Next state:\n{next_state.view(height, width)}")
    #print(f"Is terminal: {is_terminal}")
    #print(f"Value of clicked cell: {state[0][action]}")

    reward = reward_function(state, action, next_state, is_terminal)

    episode_reward += reward.item()
    memory.push(state, action, reward, next_state, is_terminal)

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

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        is_terminal_batch = torch.tensor(batch.is_terminal).int().unsqueeze(1).to(device)
        Q_values = main_network(state_batch).gather(1, action_batch)

        target_Q_values = reward_batch + GAMMA * target_network(next_state_batch).max(dim=1, keepdim=True).values * (1 - is_terminal_batch)
        
        loss = torch.nn.functional.smooth_l1_loss(Q_values, target_Q_values)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        for param in main_network.parameters():
            param.grad.data.clamp_(-10, 10)
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
            print(f"Episode {n_episodes + 1}, Loss: {mean_loss:.4f}, Moves used: {mean_moves_used:.2f}, Epsilon: {epsilon:.3f}, Episode reward: {mean_episode_reward:.2f}")
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
