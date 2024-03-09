import itertools
import random
from typing import Iterator
import numpy as np

C_EMPTY = 0
C_BLACK = 1
C_WHITE = 2

COLORNAMES = {C_EMPTY: "EMPTY", C_BLACK: "BLACK", C_WHITE: "C_WHITE"}

def next_color(color):
    return 3 - color

class PassClass:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(PassClass, cls).__new__(cls)
        return cls.__instance

    def __str__(self):
        return "PASS"


PASS = PassClass()


class BoardState:
    __slots__ = ['current', 'gameover', 'winner', 'passes', 'board']
    _DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    _EMPTY_BOARD = np.full((8, 8), C_EMPTY, dtype=np.uint8)
    _EMPTY_BOARD[3, 3] = C_WHITE
    _EMPTY_BOARD[3, 4] = C_BLACK
    _EMPTY_BOARD[4, 3] = C_BLACK
    _EMPTY_BOARD[4, 4] = C_WHITE

    def __init__(self):
        self.reset()

    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.current = self.current
        result.gameover = self.gameover
        result.winner = self.winner
        result.passes = self.passes
        result.board = self.board.copy()
        return result

    def __str__(self):
        map = {C_EMPTY: " . ", C_BLACK: " X ", C_WHITE: " O "}
        lines = []
        for row in range(8):
            line = []
            for col in range(8):
                line.append(map[self.board[row, col]])
            lines.append("".join(line))
        if not self.gameover:
            lines.append(f"| NEXT: {COLORNAMES[self.current]}")
        else:
            winner = COLORNAMES[self.winner] if self.winner is not None else "TIED"
            lines.append(f"| WINNER: {winner}")
        return "\n".join(lines)

    def reset(self):
        self.winner = None
        self.gameover = False
        self.current = C_BLACK
        self.passes = 0
        self.board = self._EMPTY_BOARD.copy()

    @staticmethod
    def inbounds(row, col):
        return (0 <= row < 8) and (0 <= col < 8)

    def move(self, move):
        if move is PASS:
            self.passes += 1
            self.current = next_color(self.current)
            if self.passes == 2:
                self._finish_and_score()
            return
        row, col = move
        self.passes = 0
        other = next_color(self.current)
        self.board[row, col] = self.current

        def direct(dr, dc):  # delta row, delta col, play move
            cr, cc = row + dr, col + dc  # current row, current col
            if not self.inbounds(cr, cc):
                return
            if self.board[cr, cc] != other:
                return
            while self.board[cr, cc] == other:
                cr, cc = cr + dr, cc + dc
                if not self.inbounds(cr, cc):
                    return
            if self.board[cr, cc] != self.current:
                return
            cr, cc = row + dr, col + dc
            while self.board[cr, cc] == other:
                self.board[cr, cc] = self.current
                cr, cc = cr + dr, cc + dc

        for dr, dc in itertools.product((-1, 0, 1), repeat=2):
            if dr == dc == 0:
                continue
            direct(dr, dc)
        self.current = other

    def _is_legal(self, row: int, col: int) -> bool:
        #if self.board[row, col] != C_EMPTY:
        #    return False
        other = next_color(self.current)

        def direct(dr, dc):  # delta row, delta col, play move
            cr, cc = row + dr, col + dc  # current row, current col
            if not self.inbounds(cr, cc):
                return False
            if self.board[cr, cc] != other:
                return False
            while self.board[cr, cc] == other:
                cr, cc = cr + dr, cc + dc
                if not self.inbounds(cr, cc):
                    return False
            if self.board[cr, cc] != self.current:
                return False
            return True

        for dr, dc in self._DELTAS:
            if direct(dr, dc):
                return True
        return False

    def gen_legal(self) -> Iterator[tuple[int, int]]:
        other = next_color(self.current)

        def gen_neighbors(row, col):
            for dr, dc in self._DELTAS:
                nr, nc = row + dr, col + dc
                if self.inbounds(nr, nc):
                    yield nr, nc

        # We'll look for all stones of opposite color and check if neighbors are legal
        found_legal = False
        poss = self.board == C_EMPTY
        neighbors = np.zeros((8, 8), dtype=bool)
        others = self.board==other
        neighbors[1:, 1:] |= others[:-1, :-1]
        neighbors[1:, :] |= others[:-1, :]
        neighbors[1:, :-1] |= others[:-1, 1:]
        neighbors[:, 1:] |= others[:, :-1]
        neighbors[:, :-1] |= others[:, 1:]
        neighbors[:-1, 1:] |= others[1:, :-1]
        neighbors[:-1, :] |= others[1:, :]
        neighbors[:-1, :-1] |= others[1:, 1:]
        poss &= neighbors

 
        # others = np.argwhere(self.board==other)
        # neighbors = np.concatenate([others + delta for delta in self._DELTAS])
        # neighbors = np.unique(neighbors, axis=0)
        # neighbors = np.compress(  (neighbors[:, 0] >= 0)
        #                        & (neighbors[:, 0] < 8)
        #                        & (neighbors[:, 1] >= 0)
        #                        & (neighbors[:, 1] < 8), neighbors, axis=0)
        for nr, nc in zip(*poss.nonzero()):
            if self._is_legal(nr, nc):
                found_legal = True
                yield nr, nc
        if not found_legal:
            yield PASS

    def _finish_and_score(self):
        self.gameover = True
        score = {C_BLACK: 0, C_WHITE: 0, C_EMPTY: 0}
        for row, col in itertools.product(range(8), repeat=2):
            score[self.board[row, col]] += 1
        sblack, swhite = score[C_BLACK], score[C_WHITE]
        if sblack > swhite:
            self.winner = C_BLACK
        elif swhite > sblack:
            self.winner = C_WHITE
        else:
            self.winner = None


class Board:
    def __init__(self):
        self.state = BoardState()
        self.history = []

    def __str__(self):
        return str(self.state)

    def reset(self):
        self.history.clear()
        self.state.reset()

    def move(self, move):
        self.history.append(self.state)
        self.state = self.state.copy()
        self.state.move(move)

    def undo(self):
        self.state = self.history.pop()

    def gen_legal(self) -> Iterator[tuple[int, int]]:
        yield from self.state.gen_legal()


class Player:
    def choose(self, board: Board):
        raise NotImplementedError


class RandomPlayer:
    def choose(self, board: Board):
        choices = list(board.gen_legal())
        if not choices:
            return None
        return random.choice(choices)

class Game:
    def __init__(self, player_black=None, player_white=None, log=True):
        if player_black is None:
            player_black = RandomPlayer()
        if player_white is None:
            player_white = RandomPlayer()
        self.players = {C_BLACK: player_black, C_WHITE: player_white}
        self.board = Board()
        self._log = log

    def reset(self):
        self.board.reset()

    def log(self, message, **kwargs):
        if self._log:
            print(message, **kwargs)

    def play_one_turn(self) -> bool:  # return is_finished
        current = self.board.state.current
        choice = self.players[current].choose(self.board)
        self.log(f"Color {COLORNAMES[current]}: {choice}")
        self.board.move(choice)
        self.log(f"{self.board}\n")
        if self.board.state.gameover:
            return True

    def undo(self):
        self.board.undo()

    def play_game(self):
        self.board.reset()
        self.log(f"{self.board}\n")
        while not self.play_one_turn():
            pass
