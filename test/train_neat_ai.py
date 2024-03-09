import numpy as np
import time
import unittest

import itertools

# from __future__ import print_function
import os
import neat
#import visualize

import sys
sys.path.append("../")

from ludopy2.player_neat import NEATPlayer
from ludopy2.player_random import RandomPlayer
import ludopy2

def train_ai(genome1, genome2, genome3, genome4, config):
    # net1 = neat.nn.FeedForwardNetwork(genome1, config)
    # net2 = neat.nn.FeedForwardNetwork(genome2, config)
    # net3 = neat.nn.FeedForwardNetwork(genome3, config)
    # net4 = neat.nn.FeedForwardNetwork(genome4, config)
    # players_in_turnament_round = [net1, net2, net3, net4]

    g = ludopy2.Game(players=[NEATPlayer(), NEATPlayer(), NEATPlayer(), NEATPlayer()])
    there_is_a_winner = False

    genomes_in_game = [genome1, genome2, genome3, genome4]

    for neat_player_idx in range(len(g.players)):
        g.players[neat_player_idx].load_neat_net(genomes_in_game[neat_player_idx], config)

    n_moves = 0
    # start_time = time.time()
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()
        
        # NOTE: input to agent:
        # # 4 int: player_pieces (the position of the pieces on the board) = "player_pieces"
        # # 4 bits: true/false (i.e., 0 or 1) for the pieces that can move (true/false if piece 1, 2, 3, 4 can move)
        # move_pieces_idx_list = [0, 0, 0, 0]
        # for move_piece_idx in move_pieces:
        #     move_pieces_idx_list[move_piece_idx] = 1
        # # 1 int: Dice roll = "dice"
        # # 4 times 3 int => 12 int: position of enemy pieces
        # Total #inputs = 21 ints ?

        # TODO: activate nets (use player_i as idx in players_in_turnament_round list) in progress (see file player_neat.py)

        # Implements random Player Move:
        if len(move_pieces):
            # print("\nOBSERVATION: len(move_pices)!=0")
            # print(f'\tPLAYER:\t{player_i}')
            piece_to_move = g.get_player_move(player_i, dice)
            
            # print(f'\tDICE:\t{dice}')
            # print(f'\tMOVE PIECES:\t{move_pieces}')
            # print(f'\tPLAYER PIECES:\t{player_pieces}')
            # print(f'\tENEMY PIECES:\t{enemy_pieces}')
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        n_moves += 1

        # if there_is_a_winner:
        #     print(f'\n\tWINNER:\t{g.first_winner_was}')
        #     print(f'\n\tWINNING ORDER:\t{g.game_winners}')
        #     print("\nOBSERVATION:")
        #     print(f'PLAYER:\t{player_i}')
        #     print(f'DICE:\t{dice}')
        #     print(f'MOVE PIECES:\t{move_pieces}')
        #     print(f'PLAYER PIECES:\t{player_pieces}')
        #     print(f'ENEMY PIECES:\t{enemy_pieces}')
        #     print(f'PLAYER IS A WINNER:\t{player_is_a_winner}')
        
        # if there_is_a_winner:
        #     # winner_player_idx = g.first_winner_was
        #     print(f'\n\nPLAYER {g.first_winner_was} IS THE WINNER')
        #     # print(f'PLAYER PIECES:\t{g.players[winner_player_idx].get_pieces()}')
        #     # (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
        #     #  there_is_a_winner), player_i = g.get_observation()
        #     # print(f'PLAYER PIECES:\t{player_pieces}')
        #     # print(f'WINNING PLAYER IS A WINNER FLAG TEST:\t{g.players[winner_player_idx].player_winner()}')

    # end_time = time.time()
    # used_time = end_time - start_time
    # moves_per_sec = n_moves / used_time
    # print("Moves per sec:", moves_per_sec)

    # Return the winner of the game
    return g.first_winner_was


# # 2-input XOR inputs and expected outputs.
# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

# def calculate_fitness():


def eval_genomes(genomes, config):
    # for genome_id, genome in genomes:
    #     genome.fitness = 4.0
    #     net = neat.nn.FeedForwardNetwork.create(genome, config)
    #     # TODO: Change for loop to run a turnament of 4 players against eachother until an ultimate winner is found.
    #     for xi, xo in zip(xor_inputs, xor_outputs):
    #         output = net.activate(xi)
    #         # TODO: Insert play_ludo_game() here??? to get if the player was a winner or not???
    #         genome.fitness -= (output[0] - xo[0]) ** 2
    # genomes_list_idx = [0, 1, 2, 3]
    # round_count = 0

    #####
    # for genome_id, genome in genomes:
    #     genome.fitness = 0

    # genome_win_stats = np.zeros(len(genomes))
    # round_count_per_player = np.zeros(len(genomes))
    # round_count_in_turnament = 0

    # for i, (genome_id_0, genome_0) in enumerate(genomes):
    #     if i >= len(genomes) - 3:
    #         break
    #     genome_0.fitness = 0 if genome_0.fitness == None else genome_0.fitness
    #     for j, (genome_id_1, genome_1) in enumerate(genomes[i+1:]):
    #         genome_1.fitness = 0 if genome_1.fitness == None else genome_1.fitness

    #         for k, (genome_id_2, genome_2) in enumerate(genomes[i+2:]):
    #             genome_2.fitness = 0 if genome_2.fitness == None else genome_2.fitness

    #             for m, (genome_id_3, genome_3) in enumerate(genomes[i+3:]):
    #                 genome_3.fitness = 0 if genome_3.fitness == None else genome_3.fitness

    #                 winner_genome_game_idx = train_ai(genome_0, genome_1, genome_2, genome_3, config)
    #                 genomes_list_idx = [i, i+1+j, i+2+k, i+3+m]
    #                 print(genomes_list_idx)
    #                 # print(i, j, k, m)
    #                 # # print(f'WINNER GENOME IDX:\t{genomes_list_idx[winner_genome_game_idx]}')
    #                 # genomes_in_game = [genome_0 ,genome_1, genome_2, genome_3]
    #                 # genomes_in_game[winner_genome_game_idx].fitness += 1.0
    #                 # round_count += 1
    #                 round_count_in_turnament += 1
    #                 for genome_in_list in genomes_list_idx:
    #                     round_count_per_player[genome_in_list] += 1
                    
    #                 genome_win_stats[genomes_list_idx[winner_genome_game_idx]] += 1
    
    # # TODO: Check that index error in this loop is fixed!!!!
    # for i, (genome_id_0, genome_0) in enumerate(genomes):
    #     # print(f'idx0={i}')
    #     if i >= len(genomes) - 4:
    #         break
    #     genome_0.fitness = 0 if genome_0.fitness == None else genome_0.fitness
    #     for j, (genome_id_1, genome_1) in enumerate(genomes[i+1:]):
    #         # print(f'idx1={i+1+j}')
    #         genome_1.fitness = 0 if genome_1.fitness == None else genome_1.fitness

    #         for k, (genome_id_2, genome_2) in enumerate(genomes[i+2+j:]):
    #             # print(f'idx2={i+2+k}')
    #             genome_2.fitness = 0 if genome_2.fitness == None else genome_2.fitness

    #             for m, (genome_id_3, genome_3) in enumerate(genomes[i+3+j+k:]):
    #                 # print(f'idx3={i+3+m}')
    #                 genome_3.fitness = 0 if genome_3.fitness == None else genome_3.fitness

    #                 winner_genome_game_idx = train_ai(genome_0, genome_1, genome_2, genome_3, config)
    #                 genomes_list_idx = [i, i+1+j, i+2+j+k, i+3+j+k+m]
    #                 print(genomes_list_idx)
    #                 # print([i, i+1+j, i+2+j, i+3+j+k])
    #                 # print(i, j, k, m)
    #                 # # print(f'WINNER GENOME IDX:\t{genomes_list_idx[winner_genome_game_idx]}')
    #                 # genomes_in_game = [genome_0 ,genome_1, genome_2, genome_3]
    #                 # genomes_in_game[winner_genome_game_idx].fitness += 1.0
    #                 # round_count += 1
    #                 round_count_in_turnament += 1
    #                 for genome_in_list in genomes_list_idx:
    #                     round_count_per_player[genome_in_list] += 1
                    
    #                 genome_win_stats[genomes_list_idx[winner_genome_game_idx]] += 1
    
    # print(genome_win_stats)
    # print(round_count_per_player)
    # # print(genome_win_stats / round_count_per_player)
    # print(f'\n\nTOTAL #ROUNDS IN THIS TURNAMENT:\t{round_count_in_turnament}')
    # for i, (genome_id, genome) in enumerate(genomes):
    #     genome.fitness = (genome_win_stats[i] / round_count_per_player[i]) * 100.0
    

    ########################
    genome_win_stats = np.zeros(len(genomes))
    round_count_per_player = np.zeros(len(genomes))
    genome_list_idx = np.arange(0, len(genomes))

    # Loop trhough all combinations of genomes playing against eachother in a game with itertools:
    for genome_in_round_list_idx, genomes_in_round in zip(itertools.combinations(genome_list_idx, 4),itertools.combinations(genomes, 4)):
        # print(genome_in_round_list_idx)
        # print(len(genomes_in_round[0]))
        genome_id_0, genome_0 = genomes_in_round[0]
        genome_id_1, genome_1 = genomes_in_round[1]
        genome_id_2, genome_2 = genomes_in_round[2]
        genome_id_3, genome_3 = genomes_in_round[3]

        winner_genome_game_idx = train_ai(genome_0, genome_1, genome_2, genome_3, config)
        genome_win_stats[genome_in_round_list_idx[winner_genome_game_idx]] += 1
        for idx in genome_in_round_list_idx:
            round_count_per_player[idx] += 1
    
    print(genome_win_stats)
    print(round_count_per_player)
    print(f'\n\nTOTAL #ROUNDS IN THIS TURNAMENT:\t{sum(genome_win_stats)}')
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = (genome_win_stats[i] / round_count_per_player[i]) * 100.0


    # for i, (genome_id_1, genome_1) in enumerate(genomes):
    #     if i >= len(genomes) - 3:
    #         break
    #     genome_1.fitness = 0 if genome_1.fitness == None else genome_1.fittness
        
    #     genome_id_2, genome_2 = genomes[i+1]
    #     genome_2.fitness = 0 if genome_2.fitness == None else genome_2.fittness
        
    #     genome_id_3, genome_3 = genomes[i+2]
    #     genome_3.fitness = 0 if genome_3.fitness == None else genome_3.fittness

    #     genome_id_4, genome_4 = genomes[i+3]
    #     genome_4.fitness = 0 if genome_4.fitness == None else genome_4.fittness

    #     train_ai(genome_1, genome_2, genome_3, genome_4, config)
        # i += 4

    # turnament_finished = False
    # genomes_in_turnament = genomes

    # while not turnament_finished:
    #     genome_winners = []
    #     n_rounds = len(genomes_in_turnament) // 4
    #     for round in range(n_rounds):
    #         genome_id_1, genome_1 = genomes_in_turnament[round*4]
    #         genome_1.fitness = 0 if genome_1.fitness == None else genome_1.fittness
            
    #         genome_id_2, genome_2 = genomes_in_turnament[round*4 + 1]
    #         genome_2.fitness = 0 if genome_2.fitness == None else genome_2.fittness
            
    #         genome_id_3, genome_3 = genomes_in_turnament[round*4+2]
    #         genome_3.fitness = 0 if genome_3.fitness == None else genome_3.fittness

    #         genome_id_4, genome_4 = genomes_in_turnament[round*4+3]
    #         genome_4.fitness = 0 if genome_4.fitness == None else genome_4.fittness

    #         genome_winners.append(train_ai(genome_1, genome_2, genome_3, genome_4, config))
    #     # for genome_turnament_idx in range(len(genome_winners)):
    #     #     genomes_in_turnament = []
    #     #     genomes_in_turnament[genome_winners]
    #     if len(genomes_in_turnament) <= 4:
    #         turnament_finished = True


    # # Run LUDO turnament between 4 AIs at a time:
    # for round in range(n_initial_rounds):
    #     genome_id_1, genome_1 = genomes[round*4]
    #     genome_1.fitness = 0 if genome_1.fitness == None else genome_1.fittness
        
    #     genome_id_2, genome_2 = genomes[round*4 + 1]
    #     genome_2.fitness = 0 if genome_2.fitness == None else genome_2.fittness
        
    #     genome_id_3, genome_3 = genomes[round*4+2]
    #     genome_3.fitness = 0 if genome_3.fitness == None else genome_3.fittness

    #     genome_id_4, genome_4 = genomes[round*4+3]
    #     genome_4.fitness = 0 if genome_4.fitness == None else genome_4.fittness

    #     train_ai(genome_1, genome_2, genome_3, genome_4, config)



def run_neat(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    
    # # Restore population from checkpoint: 
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 50)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)
    n_test_games = 100
    neat_player_win_count = 0
    for i in range(n_test_games):
        game_winner = win_rate_best_gnome(winner, config)
        if game_winner == 0:
            neat_player_win_count += 1
    
    print(f'NEAT PLAYER WON {neat_player_win_count} OUT OF {n_test_games} GAMES AGAINST 3 RANDOM PLAYER')
    return True




# def play_LUDO_game():
#     # import ludopy
#     # import numpy as np

#     g = ludopy2.Game()
#     there_is_a_winner = False

#     n_moves = 0
#     start_time = time.time()
#     while not there_is_a_winner:
#         (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
#          there_is_a_winner), player_i = g.get_observation()
        
#         # Implements random Player Move:
#         if len(move_pieces):
#             # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
#             piece_to_move = g.get_player_move(player_i, move_pieces)
#         else:
#             piece_to_move = -1

#         _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
#         n_moves += 1

#         if there_is_a_winner:
#             print(f'\n\tWINNER:\t{g.first_winner_was}')
#             print(f'\n\tWINNING ORDER:\t{g.game_winners}')
#             print("\nOBSERVATION:")
#             print(f'PLAYER:\t{player_i}')
#             print(f'DICE:\t{dice}')
#             print(f'MOVE PIECES:\t{move_pieces}')
#             print(f'PLAYER PIECES:\t{player_pieces}')
#             print(f'ENEMY PIECES:\t{enemy_pieces}')
#             print(f'PLAYER IS A WINNER:\t{player_is_a_winner}')

#     end_time = time.time()
#     used_time = end_time - start_time
#     moves_per_sec = n_moves / used_time
#     print("Moves per sec:", moves_per_sec)

#     # print("Saving history to numpy file")
#     # g.save_hist("game_history.npz")
#     # print("Saving game video")
#     # g.save_hist_video("game_video.mp4")

#     # new_hist = g.get_hist()
#     # old_hist = [[new_hist["pieces"][i], new_hist["current_dice"][i],
#     #              new_hist["current_player"][i], new_hist["round"][i]] for i in
#     #             range(len(new_hist[list(new_hist.keys())[0]]))]
#     # new_hist_2 = {"pieces": [], "current_dice": [], "current_player": [], "round": []}
#     # for pieces, current_dice, current_player, round in old_hist:
#     #     new_hist_2["pieces"].append(pieces)
#     #     new_hist_2["current_dice"].append(current_dice)
#     #     new_hist_2["current_player"].append(current_player)
#     #     new_hist_2["round"].append(round)
#     return True

def win_rate_best_gnome(winner_genome, config):
    # net1 = neat.nn.FeedForwardNetwork(genome1, config)
    # net2 = neat.nn.FeedForwardNetwork(genome2, config)
    # net3 = neat.nn.FeedForwardNetwork(genome3, config)
    # net4 = neat.nn.FeedForwardNetwork(genome4, config)
    # players_in_turnament_round = [net1, net2, net3, net4]

    g = ludopy2.Game(players=[NEATPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()])
    there_is_a_winner = False

    # genomes_in_game = [genome1, genome2, genome3, genome4]

    # for neat_player_idx in range(len(g.players)):
    #     g.players[neat_player_idx].load_neat_net(genomes_in_game[neat_player_idx], config)
    g.players[0].load_neat_net(winner_genome, config)

    n_moves = 0
    start_time = time.time()
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()
        
        # Implements Player Move:
        if len(move_pieces):
            piece_to_move = g.get_player_move(player_i, dice)
            
            # print(f'\tDICE:\t{dice}')
            # print(f'\tMOVE PIECES:\t{move_pieces}')
            # print(f'\tPLAYER PIECES:\t{player_pieces}')
            # print(f'\tENEMY PIECES:\t{enemy_pieces}')
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        n_moves += 1

        # if there_is_a_winner:
        #     print(f'\n\nWINNER:\t{g.first_winner_was}')
        #     # print(f'\n\tWINNING ORDER:\t{g.game_winners}')
        #     print("\tOBSERVATION:")
        #     print(f'PLAYER:\t{player_i}')
        #     print(f'DICE:\t{dice}')
        #     print(f'MOVE PIECES:\t{move_pieces}')
        #     print(f'PLAYER PIECES:\t{player_pieces}')
        #     print(f'ENEMY PIECES:\t{enemy_pieces}')
        #     # print(f'PLAYER IS A WINNER:\t{player_is_a_winner}')
        
        # if there_is_a_winner:
        #     # winner_player_idx = g.first_winner_was
        #     print(f'\n\nPLAYER {g.first_winner_was} IS THE WINNER')
        #     # print(f'PLAYER PIECES:\t{g.players[winner_player_idx].get_pieces()}')
        #     # (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
        #     #  there_is_a_winner), player_i = g.get_observation()
        #     # print(f'PLAYER PIECES:\t{player_pieces}')
        #     # print(f'WINNING PLAYER IS A WINNER FLAG TEST:\t{g.players[winner_player_idx].player_winner()}')

    end_time = time.time()
    used_time = end_time - start_time
    moves_per_sec = n_moves / used_time
    print("Moves per sec:", moves_per_sec)

    # Return the winner of the game
    return g.first_winner_was


class MyTestCase(unittest.TestCase):
    def test_something(self):
        # # Determine path to configuration file. This path manipulation is
        # # here so that the script will run successfully regardless of the
        # # current working directory.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, '../ludopy2/player_neat_config.txt')
        self.assertEqual(True, run_neat(config_path))

if __name__ == '__main__':
    unittest.main()