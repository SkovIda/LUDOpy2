import neat.reporting
import numpy as np
import time
import unittest
import csv

import itertools
from collections import deque

# from __future__ import print_function
import os
import neat
import visualize
# from ..neat import visualize
import pickle

import sys
sys.path.append("../")

# import glob
# from pathlib import Path
from tqdm import tqdm

from ludopy2.player_neat import NEATPlayer
from ludopy2.player_random import RandomPlayer
import ludopy2

# def train_ai(genome1, genome2, genome3, genome4, config):
#     g = ludopy2.Game(players=[NEATPlayer(), NEATPlayer(), NEATPlayer(), NEATPlayer()])
#     g.reset(g.players)  # Need to reset the game before starting a new game (bug in game class in ludopy library?)
#     there_is_a_winner = False

#     genomes_in_game = [genome1, genome2, genome3, genome4]

#     for neat_player_idx in range(len(g.players)):
#         g.players[neat_player_idx].load_neat_net(genomes_in_game[neat_player_idx], config)

#     n_moves = 0

#     # start_time = time.time()
#     while not there_is_a_winner:
#         (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
#          there_is_a_winner), player_i = g.get_observation()
        
#         # Implements random Player Move:
#         if len(move_pieces):
#             # print("\nOBSERVATION: len(move_pices)!=0")
#             piece_to_move = g.get_player_move(player_i, dice)
            
#         else:
#             piece_to_move = -1

#         _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
#         n_moves += 1

#     # end_time = time.time()
#     # used_time = end_time - start_time
#     # moves_per_sec = n_moves / used_time
#     # print("Moves per sec:", moves_per_sec)

#     # Return the winner of the game
#     return g.first_winner_was



        
# def eval_genomes_groups_of_four(genomes, config, n_games_per_start_pos=1):
#     genome_win_stats = np.zeros(len(genomes))
#     round_count_per_player = np.zeros(len(genomes))
#     n_players_in_game = 4
#     n_games = len(genomes) - (len(genomes) % n_players_in_game)
#     print(f'#genomes {len(genomes)}\t#games {n_games}')

#     for first_genome_idx in tqdm(range(0, n_games, n_players_in_game)):
#         # print(f'1st genome in game:\t {first_genome_idx}')
#         genome_id_0, genome_0 = genomes[first_genome_idx]
#         genome_id_1, genome_1 = genomes[first_genome_idx+1]
#         genome_id_2, genome_2 = genomes[first_genome_idx+2]
#         genome_id_3, genome_3 = genomes[first_genome_idx+3]

#         # genomes_in_game = genomes[first_genome_idx:first_genome_idx+n_players_in_game]
#         # genomes_in_game = [genome_0, genome_1, genome_2, genome_3]
#         genomes_in_game = [genome for genome_id,genome in genomes]

#         genomes_game_deque = deque(genomes_in_game)
#         for ith_turnament in range(n_games_per_start_pos):
#             for ith_game in range(len(genomes_in_game)):
#                 genomes_game_deque.rotate(1)
#                 # print('\n\nNEW GAME STARTED')
#                 winner_genome_game_idx = train_ai(genomes_game_deque[0], genomes_game_deque[1], genomes_game_deque[2], genomes_game_deque[3], config)
#                 genome_win_stats[first_genome_idx + winner_genome_game_idx] += 1
#                 for idx in range(first_genome_idx, first_genome_idx + 4):
#                     round_count_per_player[idx] += 1
    
#     print(genome_win_stats)
#     print(round_count_per_player)
#     print(f'\n\nTOTAL #ROUNDS IN THIS TURNAMENT:\t{sum(genome_win_stats)}')

#     for i, (genome_id, genome) in enumerate(genomes):
#         genome.fitness = (genome_win_stats[i] / round_count_per_player[i]) * 100.0




def play_AI_vs_3random(genome, neat_player_idx, config):

    players_in_game = [RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()]
    players_in_game[neat_player_idx] = NEATPlayer()
    g = ludopy2.Game(players=players_in_game)
    there_is_a_winner = False

    g.reset(g.players)  # Need to reset the game before starting a new game (bug in game class in library?)

    # genomes_in_game = [genome1, genome2, genome3, genome4]

    g.players[neat_player_idx].load_neat_net(genome, config)

    n_moves = 0
    # start_time = time.time()
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        # Implements Player Move:
        if len(move_pieces):
            piece_to_move = g.get_player_move(player_i, dice)
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        n_moves += 1

    # Return the winner of the game
    return g.first_winner_was


def eval_genomes_vs_3random(genomes, config, n_eval_games=25):
    genome_win_stats = np.zeros(len(genomes))
    round_count_per_player = np.zeros(len(genomes))

    # Loop through all genomes:
    for genome_idx, (genome_id, genome) in enumerate(tqdm(genomes)):
        # print(f'\nGENOME #{genome_idx}')
        for neat_player_start_pos in range(4):
            # print(f'\nNEAT PLAYER START POS:\t{neat_player_start_pos}')
            for game_i in range(n_eval_games):
                winner_player_idx = play_AI_vs_3random(genome, neat_player_start_pos, config)
                # print(f'WINNING PLAYER:\t{winner_player_idx}')
                if winner_player_idx == neat_player_start_pos:
                    genome_win_stats[genome_idx] += 1
                round_count_per_player[genome_idx] += 1
    
    print(f'GENOME WIN STATS:\t{genome_win_stats}')
    print(f'ROUND COUNT PER GENOME:\t{set(round_count_per_player)}')
    print(f'\n\nTOTAL #ROUNDS IN THIS TURNAMENT:\t{sum(round_count_per_player)}')

    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = (genome_win_stats[i] / round_count_per_player[i]) * 100.0



# def run_neat(config_file, winner_file, train_data_path):
def run_neat(train_data_path):
    config_file = train_data_path + 'neat_config.txt'
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    
    # # Restore population from checkpoint: 
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-19')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='neat_training_checkpoints/checkpoint_gen_'))
    # p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='neat_training_checkpoints_fitness_mean30_popsize32/checkpoint_gen_'))
    checkpoint_filename_prefix = 'checkpoint_gen_'
    p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=train_data_path + checkpoint_filename_prefix))

    # Run for up to 100 generations.
    # winner = p.run(eval_genomes, 100)
    winner = p.run(eval_genomes_vs_3random, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # print(p.generation)
    
    winner_file = train_data_path + 'winner_gen_' + str(p.generation) + '.pkl'
    print(f'\nsaving winner genome to {winner_file} ...', end='\t')
    with open(winner_file, "wb") as f:
        pickle.dump(winner, f)
        f.close()
    print('DONE')

    # print("post evaluating the best genome")
    # p.reporters.post_evaluate(p.config, p.population, p.species, p.best_genome)

    ######################################################################
    # # # # TODO: train NEAT Player against eachother, but evalue performance of an entire generation against random players
    # # checkpoint_dir = 'neat_training_checkpoints'

    # # # n_checkpoints = len(os.scandir(checkpoint_dir))
    
    # # # generation_win_stats = {}
    # # # node_names = {0: 'PIECE_0', 1: 'PIECE_1', 2: 'PIECE_2', 3: 'Piece_3'}
    # # visualize.draw_net(config, winner, True, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True, filename=train_data_path+'avg_fitness.svg')
    visualize.plot_species(stats, view=True, filename=train_data_path+'speciation.svg')

    # # highest_fitness_genome = p.best_genome

    # highest_fitness_genome = None

    # # Load data from random_ludo_game dataset:
    # assert Path('neat_training_checkpoints/').exists()
    # filepaths = glob.glob('neat_training_checkpoints/checkpoint_gen_*')

    # for i, filepath in enumerate(tqdm(filepaths)):
    #     print(filepath)
    #     p = neat.Checkpointer.restore_checkpoint(filepath)
    #     # stats.post_evaluate(config, p, )
    #     generation_winner = p.run(eval_genomes_vs_3random, 1)

    #     if highest_fitness_genome is not None:
    #         if generation_winner.fitness > highest_fitness_genome.fitness:
    #             highest_fitness_genome = generation_winner
    #     else:
    #         highest_fitness_genome = generation_winner

    # # # Visualize the best genome found in all generations:
    # # # visualize.draw_net(config, highest_fitness_genome, True, prune_unused=True)
    # # visualize.plot_stats(stats, ylog=False, view=True)
    # # visualize.plot_species(stats, view=True)
    ############################################################

    ########
    # # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))
    #########

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

    #####
    # n_test_games = 100
    # neat_player_win_count = 0
    # for i in range(n_test_games):
    #     game_winner = win_rate_best_gnome(winner, config)
    #     if game_winner == 0:
    #         neat_player_win_count += 1
    
    # print(f'NEAT PLAYER WON {neat_player_win_count} OUT OF {n_test_games} GAMES AGAINST 3 RANDOM PLAYER')
    #####

    n_test_games_per_start_pos = 250
    neat_player_win_count = 0
    n_start_pos = 4
    print(f'Evaluating the winner against 3 random palyers for {n_test_games_per_start_pos * n_start_pos}...')
    for start_pos in range(n_start_pos):
        for i in tqdm(range(n_test_games_per_start_pos)):
            game_winner = play_AI_vs_3random(winner, start_pos, config)
            if game_winner == start_pos:
                neat_player_win_count += 1
    
    print(f'NEAT PLAYER WON {neat_player_win_count} OUT OF {n_test_games_per_start_pos * n_start_pos} GAMES AGAINST 3 RANDOM PLAYERS')
    
    p.reporters.end_generation(p.config, p.population, p.species)

    return True


# # def dummy_function(genomes, config):
# #     print("dummy func called")
# #     return


# # def eval_neat_player(config_file):
# #     # Load configuration.
# #     config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
# #                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
# #                          config_file)
    
# #     # Restore population from checkpoint: 
# #     p = neat.Checkpointer.restore_checkpoint('neat_training_checkpoints_fittness_threshold_70_population_size_128/checkpoint_gen_7')

# #     # Add a stdout reporter to show progress in the terminal.
# #     p.add_reporter(neat.StdOutReporter(True))
# #     stats = neat.StatisticsReporter()
# #     p.add_reporter(stats)

# #     # # Display the winning genome.
# #     # print('\nBest genome:\n{!s}'.format(p.best_genome))

# #     # print(f'BEFORE PRINT LEN')
# #     # print(f'{len(p.reporters.reporters)}')

# #     # # # Add a stdout reporter to show progress in the terminal.
# #     # # p.add_reporter(neat.StdOutReporter(True))
# #     # # stats = neat.StatisticsReporter()
# #     # # p.add_reporter(stats)
# #     # # # p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='neat_training_checkpoints/checkpoint_gen_'))
# #     # # p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='neat_training_checkpoints_fittness_threshold_70_population_size_128/checkpoint_gen_'))

# #     best_neat_player = p.run(eval_genomes_vs_3random,1)

# #     # Display the winning genome.
# #     print('\nBest genome:\n{!s}'.format(best_neat_player))

# #     n_test_games_per_start_pos = 250
# #     neat_player_win_count = 0
# #     n_start_pos = 4
# #     for start_pos in range(n_start_pos):
# #         for i in tqdm(range(n_test_games_per_start_pos)):
# #             game_winner = play_AI_vs_3random(best_neat_player, start_pos, config)
# #             if game_winner == start_pos:
# #                 neat_player_win_count += 1
    
# #     print(f'NEAT PLAYER WON {neat_player_win_count} OUT OF {n_test_games_per_start_pos * n_start_pos} GAMES AGAINST 3 RANDOM PLAYERS')

# #     visualize.plot_stats(stats, ylog=False, view=True)
# #     visualize.plot_species(stats, view=True)

    
# #     return True



# # def play_LUDO_game():
# #     # import ludopy
# #     # import numpy as np

# #     g = ludopy2.Game()
# #     there_is_a_winner = False

# #     n_moves = 0
# #     start_time = time.time()
# #     while not there_is_a_winner:
# #         (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
# #          there_is_a_winner), player_i = g.get_observation()
        
# #         # Implements random Player Move:
# #         if len(move_pieces):
# #             # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
# #             piece_to_move = g.get_player_move(player_i, move_pieces)
# #         else:
# #             piece_to_move = -1

# #         _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
# #         n_moves += 1

# #         if there_is_a_winner:
# #             print(f'\n\tWINNER:\t{g.first_winner_was}')
# #             print(f'\n\tWINNING ORDER:\t{g.game_winners}')
# #             print("\nOBSERVATION:")
# #             print(f'PLAYER:\t{player_i}')
# #             print(f'DICE:\t{dice}')
# #             print(f'MOVE PIECES:\t{move_pieces}')
# #             print(f'PLAYER PIECES:\t{player_pieces}')
# #             print(f'ENEMY PIECES:\t{enemy_pieces}')
# #             print(f'PLAYER IS A WINNER:\t{player_is_a_winner}')

# #     end_time = time.time()
# #     used_time = end_time - start_time
# #     moves_per_sec = n_moves / used_time
# #     print("Moves per sec:", moves_per_sec)

# #     # print("Saving history to numpy file")
# #     # g.save_hist("game_history.npz")
# #     # print("Saving game video")
# #     # g.save_hist_video("game_video.mp4")

# #     # new_hist = g.get_hist()
# #     # old_hist = [[new_hist["pieces"][i], new_hist["current_dice"][i],
# #     #              new_hist["current_player"][i], new_hist["round"][i]] for i in
# #     #             range(len(new_hist[list(new_hist.keys())[0]]))]
# #     # new_hist_2 = {"pieces": [], "current_dice": [], "current_player": [], "round": []}
# #     # for pieces, current_dice, current_player, round in old_hist:
# #     #     new_hist_2["pieces"].append(pieces)
# #     #     new_hist_2["current_dice"].append(current_dice)
# #     #     new_hist_2["current_player"].append(current_player)
# #     #     new_hist_2["round"].append(round)
# #     return True

# def win_rate_best_gnome(winner_genome, config):
#     # net1 = neat.nn.FeedForwardNetwork(genome1, config)
#     # net2 = neat.nn.FeedForwardNetwork(genome2, config)
#     # net3 = neat.nn.FeedForwardNetwork(genome3, config)
#     # net4 = neat.nn.FeedForwardNetwork(genome4, config)
#     # players_in_turnament_round = [net1, net2, net3, net4]

#     g = ludopy2.Game(players=[NEATPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()])
#     there_is_a_winner = False


#     # genomes_in_game = [genome1, genome2, genome3, genome4]

#     # for neat_player_idx in range(len(g.players)):
#     #     g.players[neat_player_idx].load_neat_net(genomes_in_game[neat_player_idx], config)
#     g.players[0].load_neat_net(winner_genome, config)

#     n_moves = 0
#     start_time = time.time()
#     while not there_is_a_winner:
#         (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
#          there_is_a_winner), player_i = g.get_observation()
        
#         # Implements Player Move:
#         if len(move_pieces):
#             piece_to_move = g.get_player_move(player_i, dice)
            
#             # print(f'\tDICE:\t{dice}')
#             # print(f'\tMOVE PIECES:\t{move_pieces}')
#             # print(f'\tPLAYER PIECES:\t{player_pieces}')
#             # print(f'\tENEMY PIECES:\t{enemy_pieces}')
#         else:
#             piece_to_move = -1

#         _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
#         n_moves += 1

#         # if there_is_a_winner:
#         #     print(f'\n\nWINNER:\t{g.first_winner_was}')
#         #     # print(f'\n\tWINNING ORDER:\t{g.game_winners}')
#         #     print("\tOBSERVATION:")
#         #     print(f'PLAYER:\t{player_i}')
#         #     print(f'DICE:\t{dice}')
#         #     print(f'MOVE PIECES:\t{move_pieces}')
#         #     print(f'PLAYER PIECES:\t{player_pieces}')
#         #     print(f'ENEMY PIECES:\t{enemy_pieces}')
#         #     # print(f'PLAYER IS A WINNER:\t{player_is_a_winner}')
        
#         # if there_is_a_winner:
#         #     # winner_player_idx = g.first_winner_was
#         #     print(f'\n\nPLAYER {g.first_winner_was} IS THE WINNER')
#         #     # print(f'PLAYER PIECES:\t{g.players[winner_player_idx].get_pieces()}')
#         #     # (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
#         #     #  there_is_a_winner), player_i = g.get_observation()
#         #     # print(f'PLAYER PIECES:\t{player_pieces}')
#         #     # print(f'WINNING PLAYER IS A WINNER FLAG TEST:\t{g.players[winner_player_idx].player_winner()}')

#     end_time = time.time()
#     used_time = end_time - start_time
#     moves_per_sec = n_moves / used_time
#     # print("Moves per sec:", moves_per_sec)

#     # Return the winner of the game
#     return g.first_winner_was

# # TODO: Test performance for different network complexities - i.e., number of hidden layers.
# class MyTestCase(unittest.TestCase):
#     # def test_best_genome(self):
#     #     local_dir = os.path.dirname(__file__)
#     #     config_path = os.path.join(local_dir, '../ludopy2/player_neat_state_eval_config.txt')
#     #     self.assertEqual(True, eval_neat_player(config_path))

#     # def test_something(self):
#     #     # # Determine path to configuration file. This path manipulation is
#     #     # # here so that the script will run successfully regardless of the
#     #     # # current working directory.
#     #     local_dir = os.path.dirname(__file__)
#     #     config_path = os.path.join(local_dir, '../ludopy2/player_neat_config.txt')
#     #     self.assertEqual(True, run_neat(config_path))

#     def test_NEATPlayer_with_next_state_eval_func(self):
#         # # Determine path to configuration file. This path manipulation is
#         # # here so that the script will run successfully regardless of the
#         # # current working directory.
#         local_dir = os.path.dirname(__file__)
#         config_path = os.path.join(local_dir, '../ludopy2/player_neat_state_eval_config.txt')
#         self.assertEqual(True, run_neat(config_path))

if __name__ == '__main__':
    # # unittest.main()
    # local_dir = os.path.dirname(__file__)
    # # config_path = os.path.join(local_dir, 'neat_config_fitness_mean30_popsize32.txt')
    # # checkpoint_path = 'neat_training_checkpoints_fitness_mean30_popsize32/'
    # # winner_filename = 'winner_fitness_mean30_popsize32.pkl'
    # # run_neat(config_path, winner_filename)
    # train_data_path_1 = 'neat_training_fitness_mean60_popsize128/'
    # # config_filename = 'neat_config.txt'
    # # config_path = os.path.join(local_dir, train_data_path, config_filename)
    

    # # checkpoint_path = 'neat_training_checkpoints_fitness_mean30_popsize32/'
    # # winner_filename = 'winner_fitness_mean30_popsize32.pkl'
    # run_neat(train_data_path_1)
    
    ##### Train nets using different parameters:
    ## Test 0: 
    train_data_path_0 = 'neat_training_fitness_mean60_popsize64/'
    run_neat(train_data_path_0)

    # ## Test 1:
    # train_data_path_1 = 'neat_training_fitness_mean60_popsize128/'
    # run_neat(train_data_path_1)

    # ## Test 2:
    # train_data_path_2 = 'neat_training_fitness_mean60_popsize150/'
    # run_neat(train_data_path_2)

    # ## Test 3:
    # train_data_path_3 = 'neat_training_fitness_mean60_popsize256/'
    # run_neat(train_data_path_3)

    # ## Test 4: Baseline parameters (hyper parameters from neat paper)
    # train_data_path_4 = 'neat_training_fitness_mean60_popsize150_baseline_params/'
    # run_neat(train_data_path_4)

    # ## Test 5: No hidden layers, c3=0.5, and only add nodes:
    # train_data_path_5 = 'neat_training_fitness_mean60_popsize150_baseline_params_nohidden_c3_0-4_thresh_1-5_agument/'
    # run_neat(train_data_path_5)

    ## Test 6: No hidden

    ## Test 7: Train NEAT player against 3 Q-Players with best params (which are they???)

    ## Test 8: Train NEAT player against: 2 random players and 1 Q-Learning Player???

