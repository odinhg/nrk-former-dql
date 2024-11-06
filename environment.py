import pygame
from pygame import Color
import random


# Constants

SYMBOLS = ["B", "G", "P", "O"]
EMPTY = " "
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
    return 0 <= x < len(board) and 0 <= y < len(board[0])

def destroy_blocks(board, x, y):
    target_color = board[x][y]
    board[x][y] = EMPTY
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        if is_within_bounds(board, new_x, new_y) and board[new_x][new_y] == target_color:
            board = destroy_blocks(board, new_x, new_y)
    return board

def move_blocks_down(board):
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
    return [random.choices(SYMBOLS, k=width) for _ in range(height)]

# Main environment class
class Board:
    def __init__(self, width: int | None = None, height: int | None = None, filename: str | None = "board.txt"):
        if filename:
            self.board = read_board_from_file(filename)
        else:
            self.board = generate_random_board(width, height)
        self.width = len(self.board[0])
        self.height = len(self.board)

    def is_game_over(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] != EMPTY: 
                    return False
        return True
    
    def click(self, x, y):
        if self.board[x][y] != EMPTY:
            self.board = destroy_blocks(self.board, x, y)
            self.board = move_blocks_down(self.board)

    def render(self, block_size=50):
        pygame.init()
        res = (self.width * block_size, self.height * block_size)
        screen = pygame.display.set_mode(res)
        running = True

        x, y = 0, 0

        while running:
            if self.is_game_over():
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        y = max(0, y - 1)
                    if event.key == pygame.K_RIGHT:
                        y = min(self.width - 1, y + 1)
                    if event.key == pygame.K_UP:
                        x = max(0, x - 1)
                    if event.key == pygame.K_DOWN:
                        x = min(self.height - 1, x + 1)

                    if event.key == pygame.K_SPACE:
                        self.click(x, y)

            screen.fill(BG_COLOR)

            for i in range(self.height):
                for j in range(self.width):
                    color = COLORS[self.board[i][j]]
                    pygame.draw.rect(screen, color, (j * block_size, i * block_size, block_size, block_size), border_radius=20)

            pygame.draw.polygon(
                screen,
                CURSOR_COLOR,
                [
                    (y * block_size, x * block_size),
                    ((y + 1) * block_size, x * block_size),
                    ((y + 1) * block_size, (x + 1) * block_size),
                    (y * block_size, (x + 1) * block_size),
                ],
                width=5,
            )

            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    #board = Board(7, 9)
    board = Board(filename="board.txt")
    board.render()
