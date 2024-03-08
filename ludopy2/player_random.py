import numpy as np

from .player import Player

class RandomPlayer(Player):
    # def next_move(self, move_pieces):
    #     return move_pieces[np.random.randint(0, len(move_pieces))]
    def next_move(self, dice):
        move_pieces = self.get_pieces_that_can_move(dice)
        return move_pieces[np.random.randint(0, len(move_pieces))]