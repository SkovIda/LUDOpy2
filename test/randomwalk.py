import time
import unittest
import sys

import csv

sys.path.append("../")

import ludopy2
import numpy as np

def randwalk():
    # import ludopy
    # import numpy as np

    g = ludopy2.Game()
    there_is_a_winner = False

    n_moves = 0
    start_time = time.time()
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()
        
        # Implements random Player Move:
        if len(move_pieces):
            # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            # piece_to_move = g.get_player_move(player_i, move_pieces)
            piece_to_move = g.get_player_move(player_i, dice)
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        n_moves += 1

        # if there_is_a_winner:
        print(f'\n\tWINNER:\t{g.first_winner_was}')
        # print(f'\n\tWINNING ORDER:\t{g.game_winners}')
        print("\nOBSERVATION:")
        print(f'PLAYER:\t{player_i}')
        print(f'DICE:\t{dice}')
        print(f'MOVE PIECES:\t{move_pieces}')
        print(f'PLAYER PIECES:\t{player_pieces}')
        print(f'ENEMY PIECES:\t{enemy_pieces}')
        print(f'PLAYER IS A WINNER:\t{player_is_a_winner}')

        if there_is_a_winner:
            winner_player_idx = g.first_winner_was
            print(f'\n\nPLAYER {g.first_winner_was} IS THE WINNER')
            print(f'PLAYER PIECES:\t{g.players[winner_player_idx].get_pieces()}')
            # (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
            #  there_is_a_winner), player_i = g.get_observation()
            # print(f'PLAYER PIECES:\t{player_pieces}')
            print(f'WINNING PLAYER IS A WINNER FLAG TEST:\t{g.players[winner_player_idx].player_winner()}')
            

    end_time = time.time()
    used_time = end_time - start_time
    moves_per_sec = n_moves / used_time
    print("Moves per sec:", moves_per_sec)

    print("Saving history to numpy file")
    g.save_hist("game_history.npz")
    print("Saving game video")
    g.save_hist_video("game_video.mp4")

    new_hist = g.get_hist()
    old_hist = [[new_hist["pieces"][i], new_hist["current_dice"][i],
                 new_hist["current_player"][i], new_hist["round"][i]] for i in
                range(len(new_hist[list(new_hist.keys())[0]]))]
    new_hist_2 = {"pieces": [], "current_dice": [], "current_player": [], "round": []}
    for pieces, current_dice, current_player, round in old_hist:
        new_hist_2["pieces"].append(pieces)
        new_hist_2["current_dice"].append(current_dice)
        new_hist_2["current_player"].append(current_player)
        new_hist_2["round"].append(round)
    #print(new_hist_2)
    return True

def randwalk_all_players_finish(stats_filename, ghost_players=None):
    # import ludopy
    # import numpy as np

    g = ludopy2.Game(ghost_players=ghost_players)

    all_players_finished = False

    with open(stats_filename, 'a', newline='') as csvfile:
        winners_writer = csv.writer(csvfile)
    # fieldnames = ['first_place', 'second_place', 'thrid_place','last_place']
    # with open('names.csv', 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()

    n_moves = 0
    start_time = time.time()
    while not all_players_finished:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()
        
        # Implements random Player Move:
        if len(move_pieces):
            # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            piece_to_move = g.get_player_move(player_i, move_pieces)
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        n_moves += 1

        if ghost_players is None:
            all_players_finished = g.all_players_finish()
        elif len(g.get_winners_of_game()) >= ghost_players[0]:
                all_players_finished = True

        if all_players_finished:
            # # print(f'\n\tWINNER:\t{g.first_winner_was}')
            # print(f'\n\tWINNING ORDER:\t{g.game_winners}')
            # # print("\nOBSERVATION:")
            # # print(f'PLAYER:\t{player_i}')
            # # print(f'DICE:\t{dice}')
            # # print(f'MOVE PIECES:\t{move_pieces}')
            # # print(f'PLAYER PIECES:\t{player_pieces}')
            # # print(f'ENEMY PIECES:\t{enemy_pieces}')
            # # print(f'PLAYER IS A WINNER:\t{player_is_a_winner}')

            with open(stats_filename, 'a', newline='') as csvfile:
                # # winners_writer = csv.writer(csvfile, delimiter=' ',
                # #             quotechar='|', quoting=csv.QUOTE_MINIMAL)
                winners_writer = csv.writer(csvfile)
                # winners_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # winners_dict = {'first_place': g.game_winners[0], 'second_place': g.game_winners[1], 'thrid_place': g.game_winners[2], 'last_place': g.game_winners[3]}
                winners_writer.writerow(g.game_winners)

    end_time = time.time()
    used_time = end_time - start_time
    moves_per_sec = n_moves / used_time
    print("Moves per sec:", moves_per_sec)

    # print("Saving history to numpy file")
    # g.save_hist("game_history.npz")
    # print("Saving game video")
    # g.save_hist_video("game_video.mp4")

    # new_hist = g.get_hist()
    # old_hist = [[new_hist["pieces"][i], new_hist["current_dice"][i],
    #              new_hist["current_player"][i], new_hist["round"][i]] for i in
    #             range(len(new_hist[list(new_hist.keys())[0]]))]
    # new_hist_2 = {"pieces": [], "current_dice": [], "current_player": [], "round": []}
    # for pieces, current_dice, current_player, round in old_hist:
    #     new_hist_2["pieces"].append(pieces)
    #     new_hist_2["current_dice"].append(current_dice)
    #     new_hist_2["current_player"].append(current_player)
    #     new_hist_2["round"].append(round)
    # #print(new_hist_2)
    return True


def player_placement_rates(_player_turnament_placement, n_players=4):
    # # # Placement rate:
    # # turnament_placements = ['first_place', 'second_place', 'thrid_place','last_place']
    # # n_turnaments = len(_player_turnament_placement[0])
    # # turnament_placement_rates = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    # # print(n_turnaments)
    # # for place in range(len(_player_turnament_placement)):
    # #     print(f'\n{turnament_placements[place]}')
    # #     print(f'\tPLAYER 1:\t{_player_turnament_placement[place].count("0") / n_turnaments}')
    # #     print(f'\tPLAYER 2:\t{_player_turnament_placement[place].count("1") / n_turnaments}')
    # #     print(f'\tPLAYER 3:\t{_player_turnament_placement[place].count("2") / n_turnaments}')
    # #     print(f'\tPLAYER 4:\t{_player_turnament_placement[place].count("3") / n_turnaments}')
    # #     turnament_placement_rates[0][place] = _player_turnament_placement[place].count("0") / n_turnaments
    # #     turnament_placement_rates[1][place] = _player_turnament_placement[place].count("1") / n_turnaments
    # #     turnament_placement_rates[2][place] = _player_turnament_placement[place].count("2") / n_turnaments
    # #     turnament_placement_rates[3][place] = _player_turnament_placement[place].count("2") / n_turnaments

    # Placement rate:
    turnament_placements = ['FIRST', 'SECOND', 'THIRD', 'LAST']
    n_turnaments = len(_player_turnament_placement[0])
    turnament_stats = np.zeros((n_players,4), dtype=np.float64)
    print(n_turnaments)
    for place in range(len(_player_turnament_placement)):
        for i in range(n_players):
            turnament_stats[i][place] = _player_turnament_placement[place].count(str(i)) / n_turnaments


    # print(f'TEST #GAMES:\t{len(_player_turnament_placement[0])}\t{len(_player_turnament_placement[1])}\t{len(_player_turnament_placement[2])}\t{len(_player_turnament_placement[3])}')
    # print(f'TEST sum(%)=100:\t{np.sum(turnament_stats[0])}\t{np.sum(turnament_stats[1])}\t{np.sum(turnament_stats[2])}\t{np.sum(turnament_stats[3])}')
    print(f'TEST sum(%)=100:\t{np.sum(turnament_stats, axis=0)}')

    turnament_stats = np.round(turnament_stats, decimals=5)
    print(f'\n\t\t\t{turnament_placements}')
    for i in range(n_players):
        print(f'\tPLAYER {i}:\t{turnament_stats[i]}')
    return



class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, randwalk())

    # def test_start_pos_influence_4players(self, n_tests=3000):
    #     test_4playergame_filename = 'test_start_pos_influence_4players.csv'
    #     print('\n\nTESTING 4 PLAYER GAME:\n')
    #     for i in range(n_tests):
    #         self.assertEqual(True, randwalk_all_players_finish(test_4playergame_filename))

    #     with open(test_4playergame_filename, 'r', newline='') as csvfile:
    #         win_reader = csv.reader(csvfile)
    #         player_turnament_placement = [[], [], [], []]
    #         for row in win_reader:
    #             player_turnament_placement[0].append(row[0])
    #             player_turnament_placement[1].append(row[1])
    #             player_turnament_placement[2].append(row[2])
    #             player_turnament_placement[3].append(row[3])
    #         player_placement_rates(player_turnament_placement)
    


# TODO: Test influence of starting order on winning for 4 random players, 3 random players, 2 random players
# TODO: Test influence of starting position (incl. starting order) on winning for 3 random players

# NOTE: Runs test from terminal: python3 -m unittest -v randomwalk.py
# From unittest doc: Passing the -v option to your test script will instruct unittest.main() to enable a higher level of verbosity
if __name__ == '__main__':
    unittest.main()