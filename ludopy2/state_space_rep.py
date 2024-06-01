import numpy as np
from ludopy2 import player#, get_enemy_at_pos, BORD_TILES, TAILE_GOAL, 

# TODO: Make a representation of the LUDO board that transform an enemies pieces into the view from the current player:
# # representation should contain:
# # # The positions of each of the opponents' pieces
# # # Fields on the board that each opponent can reach before it becomes this players turn again (remember to account for the posibility of an opponent being killed before it becomes their turn)


# TODO: Check if there are errors in the calculation of enemy positions in the view from this player!!!
# # Make bitmasks: SafeSquares, Danger squares, KillSquares, AttackSquares ???
# # Make bitmasks of the positions that each of the enemies' pieces are "attacking" right now.
# # Shift this mask to calculate the danger zones for the current player
# # NOTE: If an enemy can reach another enemy before it becomes the first enemies turn, there is a risk of that enemy attacking their own globe before it becomes this player's turn again.

class PieceState:
    def __init__(self):
        self.can_activate = False
        self.can_goal = False
        self.can_kill = False
        self.can_star = False
        self.will_be_danger = False
        self.will_be_killed = False
        self.will_be_home_streatch = False
        self.will_be_safe = False

        self.will_avoid_danger = False
    
    def piece_state_as_np_array(self):
        # if self.will_be_killed or self.will_be_danger:
        #     self.will_be_safe = False
        return np.array([self.can_activate, self.can_goal, self.can_kill, self.can_star, self.will_be_danger, self.will_be_killed, self.will_be_home_streatch, self.will_be_safe])
    
    def print_piece_next_state(self):
        # print('MOVE PIECE EFFECT:')
        print(f'\tcan_activate: {self.can_activate}\n\tcan_goal {self.can_goal}\n\tcan_kill {self.can_kill}\n\tcan_star {self.can_star}\n\twill_be_danger {self.will_be_danger}\n\twill_be_killed {self.will_be_killed}\n\twill_be_home_streatch {self.will_be_home_streatch}\n\twill_be_safe {self.will_be_safe}')
    
    # def set_can_activate(self, _can_activate):
    #     self.can_activate = _can_activate

    # def set_can_goal(self, _can_goal):
    #     self.can_goal = _can_goal
    
    # def set_can_kill(self, _can_kill):
    #     self.can_kill = _can_kill
    
    # def set_can_star(self, _can_star):
    #     self.can_star = _can_star


class StateSpace:
    def __init__(self):
        self.enemy_pieces = []
        self.state = []

    def get_state(self, die, player_pieces, enemy_pieces):
        self.enemy_pieces = np.array(enemy_pieces)
        player_pieces = np.array(player_pieces)

        # print(f'\tDICE:\t{die}')
        # print(f'\tPLAYER PIECES:\t{player_pieces}')
        # print(f'\tENEMY PIECES:\t{enemy_pieces}')

        for piece in player_pieces:
            self.state.append(self.canActivate(die, piece))
            self.state.append(self.canGoal(die, piece))
            self.state.append(self.canKill(die, piece))
            self.state.append(self.canStar(die, piece))
            self.state.append(self.willbeSection1(die, piece))
            self.state.append(self.willbeSection2(die, piece))
            self.state.append(self.willbeSection3(die, piece))
            self.state.append(self.willbeSection4(die, piece))
            self.state.append(self.willbeDanger(die, piece))
            self.state.append(self.willbeKilled(die, piece))
            self.state.append(self.willbeHomeStretch(die, piece))
            self.state.append(self.willbeSafe(die, piece))

        # return np.array(self.state)   # Right formatting for ANN input
        # np.array(self.state,(4,12))
        return np.resize(
            np.array(self.state), (48, 1)
        )  # Right formatting for ANN input
        

    ### Oppotunities ###
    def canActivate(self, die, piece_position):
        if piece_position != 0:
            return False
        if die == 6:
            return True
        else:
            return False
        # if die == 6 and piece_position == 0:
        #     return True
        # return False

    def canGoal(self, die, piece_position):
        position = piece_position + die
        if position == player.STAR_AT_GOAL_AREAL_INDX:
            return True
        if position == player.GOAL_INDEX:
            return True
        else:
            return False

    def canKill(self, die, piece_position):
        if self.willbeKilled(die, piece_position):
            return False
        differences = self.enemy_pieces - piece_position
        kill = np.any(differences == die)
        return kill

    def canStar(self, die, piece_position):
        position = piece_position + die
        if position == player.STAR_AT_GOAL_AREAL_INDX:
            return True
        star = np.any(np.array(player.STAR_INDEXS) == position)
        return star

    ### Consequences ###

    def willbeSection1(self, die, piece_position):
        section = range(1, 14)
        return (piece_position + die) in section

    def willbeSection2(self, die, piece_position):
        section = range(14, 27)
        return (piece_position + die) in section

    def willbeSection3(self, die, piece_position):
        section = range(27, 40)
        return (piece_position + die) in section

    def willbeSection4(self, die, piece_position):
        section = range(40, 53)
        return (piece_position + die) in section

    def willbeDanger(self, die, piece_position):
        if self.willbeSafe(die, piece_position) == True:
            return False
        if self.willbeKilled(die, piece_position) == True:
            return True

        position = piece_position + die

        enemy_globes = [
            player.ENEMY_1_GLOB_INDX,
            player.ENEMY_2_GLOB_INDX,
            player.ENEMY_3_GLOB_INDX,
        ]
        if np.any(np.array(enemy_globes) == position):
            return True

        if position > 6:
            differences = (position) - self.enemy_pieces
            killzone = ((0 < differences) & (differences <= 6)).any()
        else:  # TODO: Fix! (wrap around - ex. 53 is a danger to 1-6)
            differences = (position) - self.enemy_pieces
            killzone = ((0 < differences) & (differences <= 6)).any()

        return killzone

    def willbeKilled(self, die, piece_position):
        # TODO: land on enemy home globe with correct enemy

        # Check for duplicates on one field
        unique, counts = np.unique(self.enemy_pieces, return_counts=True)
        lookup = dict(zip(unique, counts))
        try:
            return lookup[piece_position + die] >= 2
        except:
            return False

    def willbeHomeStretch(self, die, piece_position):
        if piece_position + die >= 53:
            return True
        if piece_position + die == player.STAR_AT_GOAL_AREAL_INDX:
            return True
        else:
            return False

    def willbeSafe(self, die, piece_position):
        if self.willbeHomeStretch(die, piece_position) == True:
            return True
        position = piece_position + die
        safe = np.any(np.array(player.GLOB_INDEXS) == position)
        safe |= player.START_INDEX
        return safe


if __name__ == "__main__":
    state = StateSpace()

    print(
        state.get_state(
            2,
            [0, 0, 7, 30],
            np.array([[52, 3, 7, 35], [43, 3, 4, 36], [6, 8, 2, 35], [52, 3, 7, 35]]),
        )
    )
    # print(state.canStar(2, 36))