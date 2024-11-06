import pygame
import sys
import random

def click_on_block(board, x, y):
    color = board[x][y]
    board[x][y] = " "
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < len(board) and 0 <= new_y < len(board[0]) and board[new_x][new_y] == color:
            board = click_on_block(board, new_x, new_y)
    return board

def move_blocks_down(board):
    for j in range(len(board[0])):
        new_column = [block for block in [row[j] for row in board] if block != " "]
        new_column = [" "] * (len(board) - len(new_column)) + new_column
        for i, block in enumerate(new_column):
            board[i][j] = block
    return board

def is_game_over(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != " ":
                return False
    return True

pygame.init()

n_moves = 0
width = 7
height = 9
block_size = 50

res = (width * block_size, height * block_size)

colors = {
    "B": pygame.Color("#35b1e7"),
    "G": pygame.Color("#5dc479"),
    "P": pygame.Color("#f958ab"),
    "O": pygame.Color("#ff9c54"),
    " ": pygame.Color("#000000"),
}

symbols = ["B", "G", "P", "O"]
board = []
for i in range(height):
    row = random.choices(symbols, k=width)
    board.append(row)

x, y = (0, 0)

screen = pygame.display.set_mode(res)

running = True
while running:
    if is_game_over(board):
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
                y = min(width - 1, y + 1)
            if event.key == pygame.K_UP:
                x = max(0, x - 1)
            if event.key == pygame.K_DOWN:
                x = min(height - 1, x + 1)

            if event.key == pygame.K_SPACE and board[x][y] != " ":
                board = click_on_block(board, x, y)
                board = move_blocks_down(board)
                n_moves += 1

    # Draw to screen
    screen.fill(pygame.Color("black"))

    for i in range(height):
        for j in range(width):
            color = colors[board[i][j]]
            pygame.draw.rect(screen, color, (j * block_size, i * block_size, block_size, block_size), border_radius=20)

    pygame.draw.polygon(
        screen,
        pygame.Color("red"),
        [
            (y * block_size, x * block_size),
            ((y + 1) * block_size, x * block_size),
            ((y + 1) * block_size, (x + 1) * block_size),
            (y * block_size, (x + 1) * block_size),
        ],
        width=5,
    )

    pygame.display.set_caption(f"Moves: {n_moves}")

    pygame.display.flip()

pygame.quit()
