import torch

from model import Model
from environment import Board


board_file = "board.txt"
model_file = "main_network.pt"

board = Board(filename=board_file)
width, height = board.width, board.height

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
main_network = Model(width, height)
main_network.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))

main_network.eval()

is_terminal = False

n_moves = 0

while not is_terminal:
    state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)
    valid_actions = state.view(-1).nonzero().squeeze(1)
    state = state.to(device)
    q_values = main_network(state).view(-1)
    valid_q_values = q_values.cpu()[valid_actions]
    action_idx = valid_q_values.argmax()
    action = valid_actions[action_idx].item()
    board.click(action)
    next_state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)
    is_terminal = board.is_game_over()
    print(f"State:\n{state.view(height, width)}")
    print(f"Action: {action} (x={action % width}, y={action // width})")
    n_moves += 1

print(f"Board cleared in {n_moves} moves.")
