# 🧩 Solving NRK's "Former" Puzzle Game Using Deep Q-Networks

This repository contains the code for solving [NRK's "Former"](https://www.nrk.no/former-1.17105310) puzzle game using Deep Q-Learning. The game is a 2D grid-based puzzle game where the goal is to remove all the blocks using as few moves as possible. The game is played on a 9x7 grid where each cell can contain a block of one of the four colors: pink, green, blue, and orange. When a block is clicked, all blocks of the same color that are connected are also removed. The game is won when all blocks are removed.

![Game Screenshot](./figs/nrk_screenshot.png)

**Figure 1:** Screenshot of the game from NRK's website.

## 🎮 Playing the Game 

The game is re-implemented in Python using Pygame. To play the game yourself, run `python environment.py`. The game is played using the arrow keys to move the cursor and the space bar to remove blocks. The game is won when all blocks are removed.

The board is loaded from the file `board.txt` which contains the initial state of the board. The board is a 9x7 grid where each cell contains a block of one of the four colors. The colors are represented by the following characters: pink [P], green [G], blue [B], and orange [O].

## 🤖 Solving the Game Using Deep Q-Learning 

Run `python inference.py` to let the agent solve the game using the trained model. The agent will play the game using the best action according to the trained Q-values. A screenshot of each step is saved in the `./policy_solution` directory.

![Agent Playing the Game](./figs/animation.gif)

The board to solve is loaded from the file `board.txt` and can be changed to any board configuration.

## 🧠 Training the Model

The agent is trained using Deep Q-Networks (DQN) with experience replay and target network. To train the model from scratch, delete the model checkpoint `main_network.pt` and run `python dqn.py`. The neural network is specified in `model.py`. Hyper-parameter settings can be changed in `dqn.py`.

## 📝 Notes

The main goal of this project was for me to learn about DQN. The code can be improved in many ways, and the agent can be trained to solve the game more efficiently.

It would be interesting to implement a more advanced variation of DQN, such as, Double DQN, Dueling DQN, or Prioritized Experience Replay to see if the agent can learn to solve the game more efficiently.
