import pygame
from pygame import Color
import random


# Constants
SYMBOLS = ["B", "G", "P", "O"]
EMPTY = " "
SYMBOL_TO_INDEX = {symbol: i for i, symbol in enumerate([EMPTY] + SYMBOLS)}
COLORS = {
    "B": Color("#35b1e7"),
    "G": Color("#5dc479"),
    "P": Color("#f958ab"),
    "O": Color("#ff9c54"),
    " ": Color("#000000"),
} 
BG_COLOR = Color("#000000")
CURSOR_COLOR = Color("#ff0000")

# Helper functions
def is_within_bounds(board, x, y):
    return 0 <= y < len(board) and 0 <= x < len(board[0])

def destroy_blocks(board, x, y):
    # Recursively destroy blocks of the same color
    target_color = board[y][x]
    board[y][x] = EMPTY
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        if is_within_bounds(board, new_x, new_y) and board[new_y][new_x] == target_color:
            board = destroy_blocks(board, new_x, new_y)
    return board

def move_blocks_down(board):
    # Move blocks down to fill empty spaces
    for j in range(len(board[0])):
        new_column = [block for block in [row[j] for row in board] if block != EMPTY]
        new_column = [EMPTY] * (len(board) - len(new_column)) + new_column
        for i, block in enumerate(new_column):
            board[i][j] = block
    return board

def read_board_from_file(filename):
    with open(filename, "r") as f:
        board = [list(line.strip()) for line in f]

    if not all(all(block in SYMBOLS for block in row) for row in board):
        raise ValueError("Invalid board file: contains invalid symbols")

    if not all(len(row) == len(board[0]) for row in board):
        raise ValueError("Invalid board file: rows have different lengths")

    return board

def generate_random_board(width, height):
    # Generate the same env everytime (for testing DQN)
    rng = random.Random()
    rng.seed(0)
    return [rng.choices(SYMBOLS, k=width) for _ in range(height)]
    #return [random.choices(SYMBOLS, k=width) for _ in range(height)]

def count_non_empty_blocks(board):
    return sum(block != EMPTY for row in board for block in row)

def number_of_blocks_removed(board1, board2): 
    return abs(count_non_empty_blocks(board1) - count_non_empty_blocks(board2))

# Main environment class
class Board:
    def __init__(self, width: int | None = None, height: int | None = None, filename: str | None = None):
        if filename:
            self.board = read_board_from_file(filename)
        else:
            self.board = generate_random_board(width, height)
        self.width = len(self.board[0])
        self.height = len(self.board)
        self.clicks = 0

    def reset(self):
        self.board = generate_random_board(self.width, self.height)
        self.clicks = 0

    def index_to_coords(self, index):
        return index % self.width, index // self.width

    def coords_to_index(self, x, y):
        return y * self.width + x

    def click(self, index):
        x, y = self.index_to_coords(index)
        if self.is_not_empty(x, y):
            self.board = destroy_blocks(self.board, x, y)
            self.board = move_blocks_down(self.board)
            self.clicks += 1

    def is_game_over(self):
        return all(all(block == EMPTY for block in row) for row in self.board)

    def is_not_empty(self, x, y):
        return self.board[y][x] != EMPTY

    def get_encoded_board(self):
        return [SYMBOL_TO_INDEX[block] for row in self.board for block in row]

    def render(self, block_size=50):
        pygame.init()
        res = (self.width * block_size, self.height * block_size)
        screen = pygame.display.set_mode(res)
        running = True

        n_moves = 0
        x, y = 0, 0

        while running:
            #if is_game_over(self.board):
            #    running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        x = max(0, x - 1)
                    if event.key == pygame.K_RIGHT:
                        x = min(self.width - 1, x + 1)
                    if event.key == pygame.K_UP:
                        y = max(0, y - 1)
                    if event.key == pygame.K_DOWN:
                        y = min(self.height - 1, y + 1)

                    if event.key == pygame.K_SPACE:
                        n_moves += 1
                        self.click(self.coords_to_index(x, y))
                        pygame.display.set_caption(f"Moves: {n_moves}")

            screen.fill(BG_COLOR)

            for i in range(self.height):
                for j in range(self.width):
                    color = COLORS[self.board[i][j]]
                    pygame.draw.rect(screen, color, (j * block_size, i * block_size, block_size, block_size), border_radius=20)

            pygame.draw.polygon(
                screen,
                CURSOR_COLOR,
                [
                    (x * block_size, y * block_size),
                    ((x + 1) * block_size, y * block_size),
                    ((x + 1) * block_size, (y + 1) * block_size),
                    (x * block_size, (y + 1) * block_size),
                ],
                width=5,
            )


            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    board = Board(3, 4)
    #board = Board(filename="board.txt")
    board.render()
