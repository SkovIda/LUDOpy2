import numpy as np
import time
import itertools

# from __future__ import print_function
import os
import neat
#import visualize

from .player import Player
import ludopy2

# TODO: Use the neat-python xor example to build a NEAT LUDO Player - see ../neat/evolve-feedforward.py
# NOTE: tutorial for training Pong player with neat-python: https://www.youtube.com/watch?v=2f6TmKm7yx0
# See the "monopoly AI video on youtube" for training (turnament setup and fitness values): https://www.youtube.com/watch?v=dkvFcYBznPI

class NEATPlayer(Player):
    def load_neat_net(self, _genome, _config):
        self.genome = _genome
        self.config = _config
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

    def next_move(self, dice, _player_pieces, _enemy_pieces):
        move_pieces = self.get_pieces_that_can_move(dice)

        if len(move_pieces) == 1:
            return move_pieces[0]

        can_move_piece_list = [0, 0, 0, 0]
        for move_piece_idx in move_pieces:
            can_move_piece_list[move_piece_idx] = 1

        # own_pieces = self.get_pieces(_player_pieces)
        own_pieces_pos = _player_pieces
        # enemy_0_pieces = _enemy_pieces[0]
        # enemy_1_pieces = _enemy_pieces[1]
        # enemy_2_pieces = _enemy_pieces[2]
        enemy_pieces_pos = list(itertools.chain.from_iterable(_enemy_pieces))

        # Input to NEAT Net:
        # # 4 int: own_pieces
        # # 4 int: (True/False = 0/1) can_move_piece_list = [0, 0, 0, 0]
        # # 12 int: pos of all 4 pieces for each enemy
        # # 1 int: dice
        # net_input = own_pieces
        # np.concatenate((net_input, can_move_piece_list), axis=None)
        enemy_pieces_pos = np.reshape(_enemy_pieces, (1,12))
        net_input = np.concatenate((dice, own_pieces_pos, can_move_piece_list, enemy_pieces_pos), axis=None)
        # net_input = np.concatenate(dice, own_pieces_pos, can_move_piece_list, enemy_pieces_pos, axis=1)
        print(type(net_input))
        print(net_input)
        
        # net_input.extend(enemy_pieces_flattened)
        # net_input.extend(dice)
        print(f'type(net_input)={type(net_input)}')
        net_output = self.net.activate(net_input)
        print(f'type(net_output)={type(net_output)}')
        # # Get the output move from the network and round to nearest legal move:
        print("NET OUTPUT:")
        print(f'\tBEFORE ACTION SELECT:\n\t{net_output}')
        next_move = min(move_pieces, key=lambda x:abs(x-net_output[0]))
        print(f'\tAFTER ACTION SELECT:\n\t{next_move}')
        return next_move #min(move_pieces, key=lambda x:abs(x-net_output[0]))

    # def next_move(self, move_pieces):
    #     net_output = self.net.activate()
    #     # Get the output move from the network and round to nearest legal move:
    #     return min(move_pieces, key=lambda x:abs(x-net_output))
    #     # return move_pieces[np.random.randint(0, len(move_pieces))]
    
    # def eval_genomes(genomes, config):
    #     for genome_id, genome in genomes:
    #         genome.fitness = 4.0
    #         net = neat.nn.FeedForwardNetwork.create(genome, config)
    #         for xi, xo in zip(xor_inputs, xor_outputs):
    #             output = net.activate(xi)
    #             genome.fitness -= (output[0] - xo[0]) ** 2



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
    start_time = time.time()
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()
        
        # NOTE: input to agent:
        # # 4 int: player_pieces (the position of the pieces on the board) = "player_pieces"
        # # 4 bits: true/false (i.e., 0 or 1) for the pieces that can move (true/false if piece 1, 2, 3, 4 can move)
        move_pieces_idx_list = [0, 0, 0, 0]
        for move_piece_idx in move_pieces:
            move_pieces_idx_list[move_piece_idx] = 1
        # # 1 int: Dice roll = "dice"
        # # 4 times 3 int => 12 int: position of enemy pieces
        # Total #inputs = 21 ints ?

        # TODO: activate nets (use player_i as idx in players_in_turnament_round list)???

        # Implements random Player Move:
        if len(move_pieces):
            # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            piece_to_move = g.get_player_move(player_i, dice)
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

    # Return the winner of the game
    return g.first_winner_was


# # 2-input XOR inputs and expected outputs.
# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

def eval_genomes(genomes, config):
    # for genome_id, genome in genomes:
    #     genome.fitness = 4.0
    #     net = neat.nn.FeedForwardNetwork.create(genome, config)
    #     # TODO: Change for loop to run a turnament of 4 players against eachother until an ultimate winner is found.
    #     for xi, xo in zip(xor_inputs, xor_outputs):
    #         output = net.activate(xi)
    #         # TODO: Insert play_ludo_game() here??? to get if the player was a winner or not???
    #         genome.fitness -= (output[0] - xo[0]) ** 2
    genomes_list_idx = [0, 1, 2, 3]

    for i, (genome_id_0, genome_0) in enumerate(genomes):
        if i >= len(genomes) - 3:
            break
        genome_0.fitness = 0 if genome_0.fitness == None else genome_0.fittness
        for j, (genome_id_1, genome_1) in enumerate(genomes[i+1:]):
            genome_1.fitness = 0 if genome_1.fitness == None else genome_1.fittness

            for k, (genome_id_2, genome_2) in enumerate(genomes[i+2:]):
                genome_2.fitness = 0 if genome_2.fitness == None else genome_2.fittness

                for m, (genome_id_3, genome_3) in enumerate(genomes[i+3:]):
                    genome_3.fitness = 0 if genome_3.fitness == None else genome_3.fittness

                    winner_genome_game_idx = train_ai(genome_0, genome_1, genome_2, genome_3, config)
                    genomes_list_idx = [i, i+1+j, i+2+k, i+3+m]
                    print(f'WINNER GENOME IDX:\t{genomes_list_idx[winner_genome_game_idx]}')
                    winner_genome_id, winner_genome = genomes[genomes_list_idx[winner_genome_game_idx]]
                    winner_genome.fittness += 1.0
                    #_in_game = [genome_1, genome_2, genome_3, genome_4]



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
    p.add_reporter(neat.Checkpointer(1))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 50)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # # visualize.draw_net(config, winner, True, node_names=node_names)
    # # visualize.plot_stats(stats, ylog=False, view=True)
    # # visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)




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



if __name__ == '__main__':
    # # Determine path to configuration file. This path manipulation is
    # # here so that the script will run successfully regardless of the
    # # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'player_neat_config.txt')
    run_neat(config_path)