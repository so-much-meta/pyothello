import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import numpy as np
from .othello import *


class Evaluator:
    """Evaluate a board state."""

    def eval(self, board_state: BoardState):
        """Output probababily of current player winning, from 0..1"""
        raise NotImplementedError


class Net1(nn.Module):
    """Simple fully connected network.

    8x8 is_black
    8x8 is_white
    1 last_pass
    1 current_black
    1 current_white
    131 total inputs"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(131, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = x
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out
    
class Net1b(nn.Module):
    """Simple fully connected network.

    8x8 is_black
    8x8 is_white
    1 last_pass
    1 current_black
    1 current_white
    131 total inputs"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(131, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        out = x
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out    


class Net2(nn.Module):
    """Simple network with convolutions

    5 input channels:
    8x8 is_black
    8x8 is_white
    8x8 (all) last_pass
    8x8 (all) current_black
    8x8 (all) current_white
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 5, 3, padding=1)
        self.conv2 = nn.Conv2d(5, 2, 3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out = x
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = out.view(-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

class Net3(nn.Module):
    """Simple network with convolutions

    5 input channels:
    8x8 is_black
    8x8 is_white
    8x8 (all) last_pass
    8x8 (all) current_black
    8x8 (all) current_white
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 1, 3, padding=1)
        self.fc1 = nn.Linear(8 * 8, 1)

    def forward(self, x):
        out = x
        out = F.relu(self.conv1(out))
        out = out.view(-1)
        out = F.sigmoid(self.fc1(out))
        return out

class NeuralEvaluator(Evaluator):
    def __init__(self, net):
        self.net = net
        weights_shape = next(net.parameters()).size()
        FLAT = len(weights_shape) == 2  # fully connected
        self.load = None
        if FLAT:
            inputs = weights_shape[1]
            if inputs == 131:
                self.input_tensor = torch.FloatTensor(131)
                self.load = self.load_flat_131
        else:
            input_channels = weights_shape[1]
            if input_channels == 5:
                self.input_tensor = torch.FloatTensor(5, 8, 8)
                self.load = self.load_conv_5
        if not self.load:
            raise RuntimeError(f"Invalid weights shape: {weights_shape}")

    def load_flat_131(self, board_state: BoardState):  # Net1 type model
        black_idx = 0
        white_idx = 64
        for row, col in itertools.product(range(8), repeat=2):
            color = board_state.board[row][col]
            if color == Color.BLACK:
                self.input_tensor[black_idx] = 1
                self.input_tensor[white_idx] = 0
            elif color == Color.WHITE:
                self.input_tensor[black_idx] = 0
                self.input_tensor[white_idx] = 1
            else:
                self.input_tensor[black_idx] = 0
                self.input_tensor[white_idx] = 0
            black_idx += 1
            white_idx += 1
        self.input_tensor[128] = 1 if board_state.passes else 0
        self.input_tensor[129] = 1 if board_state.current == Color.BLACK else 0
        self.input_tensor[130] = 1 if board_state.current == Color.WHITE else 0

    def load_conv_5(self, board_state: BoardState):
        for row, col in itertools.product(range(8), repeat=2):
            color = board_state.board[row][col]
            if color == Color.BLACK:
                self.input_tensor[0][row][col] = 1
                self.input_tensor[1][row][col] = 0
            elif color == Color.WHITE:
                self.input_tensor[0][row][col] = 0
                self.input_tensor[1][row][col] = 1
            else:
                self.input_tensor[0][ row][col] = 0
                self.input_tensor[1][row][col] = 0
            self.input_tensor[2][row][col] = 1 if board_state.passes else 0
            self.input_tensor[3] = 1 if board_state.current == Color.BLACK else 0
            self.input_tensor[4] = 1 if board_state.current == Color.WHITE else 0

    def raw_eval(self, board_state: BoardState):
        self.load(board_state)
        return self.net(self.input_tensor)

    def eval(self, board_state: BoardState):
        result = self.raw_eval(board_state)
        return result.item() if board_state.current == Color.BLACK else 1 - result.item()


class NeuralPlayer(Player):
    def __init__(self, evaluator, temperature=None):
        self.temperature = temperature
        self.evaluator = evaluator
        self._log = False

    def choose(self, board: Board):
        eval_choices = []
        for choice in board.gen_legal():
            if choice is PASS and (not self._log or board.state.passes):
                return choice
            board.move(choice)
            eval = 1 - self.evaluator.eval(board.state)  # minimize next player's eval
            eval_choices.append((eval, choice))
            board.undo()
        if self.temperature is None:
            eval, choice = max(eval_choices)
            if self._log:
                print(f"Chose {choice} with eval {eval}")
            return choice
        evals, choices = zip(*eval_choices)
        arr = np.log(evals) / self.temperature
        arr = np.exp(arr)
        arr = arr / arr.sum()
        result_idx = np.random.choice(len(arr), p=arr)
        choice = choices[result_idx]
        eval = evals[result_idx]
        if self._log:
            print(f"Chose {choice} with eval {eval}")
        return choice


class Trainer:
    def __init__(self):
        # self.net = Net1()
        # self.net = Net3()
        self.net = Net1b()
        self.evaluator = NeuralEvaluator(self.net)
        self.player = NeuralPlayer(self.evaluator, temperature=0.5)
        self.game = Game(player_black=self.player, player_white=self.player, log=False)
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)

    def play_training_game(self):
        self.net.eval()
        self.game.play_game()

    def play_sample_game(self):
        _temp = self.player.temperature
        self.player.temperature = None
        self.net.eval()
        self.game._log = True
        self.player._log = True
        self.game.play_game()
        self.game._log = False
        self.player._log = False
        self.player.temperature = _temp

    def learn_from_game(self):
        self.net.train()
        winner = self.game.board.state.winner
        if winner == Color.BLACK:
            y_val = 1.0
        elif winner == Color.WHITE:
            y_val = 0.0
        else:
            y_val = 0.5
        for state in self.game.board.history:
            y_pred = self.evaluator.raw_eval(state)
            loss = self.loss_fn(y_pred, torch.tensor([y_val]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
