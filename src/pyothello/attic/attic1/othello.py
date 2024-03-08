import enum
import itertools
import random
from typing import Iterator
import copy


class Color(enum.Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def next(self):
        return Color(3 - self.value)


C = Color


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
    _DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    _EMPTY_BOARD = [[C.EMPTY] * 8 for _ in range(8)]
    _EMPTY_BOARD[3][3] = C.WHITE
    _EMPTY_BOARD[3][4] = C.BLACK
    _EMPTY_BOARD[4][3] = C.BLACK
    _EMPTY_BOARD[4][4] = C.WHITE

    def __init__(self):
        self.reset()

    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.current = self.current
        result.gameover = self.gameover
        result.winner = self.winner
        result.passes = self.passes
        result.board = [l[:] for l in self.board]
        return result

    def __str__(self):
        map = {C.EMPTY: " ", C.BLACK: "#", C.WHITE: "O"}
        lines = []
        for row in range(8):
            line = []
            for col in range(8):
                line.append(map[self.board[row][col]])
            lines.append("".join(line))
        if not self.gameover:
            lines.append(f"NEXT: {self.current.name}")
        else:
            winner = self.winner.name if self.winner is not None else "TIED"
            lines.append(f"WINNER: {winner}")
        return "\n".join(lines)

    def reset(self):
        self.winner = None
        self.gameover = False
        self.current = C.BLACK
        self.passes = 0
        self.board = [l[:] for l in self._EMPTY_BOARD]

    @staticmethod
    def inbounds(row, col):
        return (0 <= row < 8) and (0 <= col < 8)

    def move(self, move):
        if move is PASS:
            self.passes += 1
            self.current = self.current.next()
            if self.passes == 2:
                self._finish_and_score()
            return
        row, col = move
        self.passes = 0
        other = self.current.next()
        self.board[row][col] = self.current

        def direct(dr, dc):  # delta row, delta col, play move
            cr, cc = row + dr, col + dc  # current row, current col
            if not self.inbounds(cr, cc):
                return
            if self.board[cr][cc] != other:
                return
            while self.board[cr][cc] == other:
                cr, cc = cr + dr, cc + dc
                if not self.inbounds(cr, cc):
                    return
            if self.board[cr][cc] != self.current:
                return
            cr, cc = row + dr, col + dc
            while self.board[cr][cc] == other:
                self.board[cr][cc] = self.current
                cr, cc = cr + dr, cc + dc

        for dr, dc in itertools.product((-1, 0, 1), repeat=2):
            if dr == dc == 0:
                continue
            direct(dr, dc)
        self.current = other

    def is_legal(self, row: int, col: int) -> bool:
        if self.board[row][col] != C.EMPTY:
            return False
        other = self.current.next()

        def direct(dr, dc):  # delta row, delta col, play move
            cr, cc = row + dr, col + dc  # current row, current col
            if not self.inbounds(cr, cc):
                return False
            if self.board[cr][cc] != other:
                return False
            while self.board[cr][cc] == other:
                cr, cc = cr + dr, cc + dc
                if not self.inbounds(cr, cc):
                    return False
            if self.board[cr][cc] != self.current:
                return False
            return True

        for dr, dc in self._DELTAS:
            if dr == dc == 0:
                continue
            if direct(dr, dc):
                return True
        return False

    def gen_legal(self) -> Iterator[tuple[int, int]]:
        other = self.current.next()

        def gen_neighbors(row, col):
            for dr, dc in self._DELTAS:
                nr, nc = row + dr, col + dc
                if self.inbounds(nr, nc):
                    yield nr, nc

        # We'll look for all stones of opposite color and check if neighbors are legal
        visited = set()
        found_legal = False
        for row, col in itertools.product(range(8), repeat=2):
            if self.board[row][col] != other:
                continue
            for nr, nc in gen_neighbors(row, col):
                if (nr, nc) in visited:
                    continue
                visited.add((nr, nc))
                if self.is_legal(nr, nc):
                    found_legal = True
                    yield nr, nc
        if not found_legal:
            yield PASS

    def _finish_and_score(self):
        self.gameover = True
        score = {C.BLACK: 0, C.WHITE: 0, C.EMPTY: 0}
        for row, col in itertools.product(range(8), repeat=2):
            score[self.board[row][col]] += 1
        sblack, swhite = score[C.BLACK], score[C.WHITE]
        if sblack > swhite:
            self.winner = C.BLACK
        elif swhite > sblack:
            self.winner = C.WHITE
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
        self.players = {C.BLACK: player_black, C.WHITE: player_white}
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
        self.log(f"Color {current.name}: {choice}")
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
