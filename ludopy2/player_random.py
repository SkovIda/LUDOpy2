from .player import Player
#import ludopy
import numpy as np

class RandomPlayer(Player):
    def next_move(self, move_pieces):
        return move_pieces[np.random.randint(0, len(move_pieces))]