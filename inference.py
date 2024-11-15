import torch
from pathlib import Path
from model import Model
from environment import Board

board_file = "board.txt"
model_file = "main_network.pt"
output_dir = Path("policy_solution")

for f in output_dir.glob("*.png"):
    f.unlink()

board = Board(filename=board_file)
width, height = board.width, board.height

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
main_network = Model(width, height)
main_network.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
main_network.to(device)
main_network.eval()

is_terminal = False
actions = []

while not is_terminal:
    state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)
    valid_actions = state.view(-1).nonzero().squeeze(1)
    state = state.to(device)
    q_values = main_network(state).view(-1)
    valid_q_values = q_values.cpu()[valid_actions]
    action_idx = valid_q_values.argmax()
    action = valid_actions[action_idx].item()
    board.save_board_image(output_dir / f"{len(actions):03}.png", action)
    board.click(action)
    next_state = torch.tensor(board.get_encoded_board()).float().unsqueeze(0)
    is_terminal = board.is_game_over()
    actions.append(action)

print(f"Board cleared in {len(actions)} moves:")
for i, action in enumerate(actions):
    print(f"{i+1}.\t({action % width}, {action // width})")
