import neat.genes
import neat.genome
import numpy as np
import time
import itertools

import os
import neat

from .player import Player
from .state_space_rep import StateSpace, PieceState
import ludopy2

# NOTE: See the "monopoly AI video on youtube" for training (turnament setup and fitness values): https://www.youtube.com/watch?v=dkvFcYBznPI

class NEATPlayer(Player):
    def load_neat_net(self, _genome, _config):
        self.genome = _genome
        self.config = _config
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        # self.abstract_state_rep = StateSpace()
        self.abstract_state_rep = np.zeros((4,8))

    # def generate_abstract_net_input(self, dice, player_pieces, enemy_pieces):
    #     # Possible actions for each piece:
    #     return np.reshape(self.abstract_state_rep.get_state(dice, player_pieces, enemy_pieces),(4,12))
    
    def generate_abstract_net_input(self, dice, player_pieces, enemy_pieces):
        # print(f'\n\nPLAYER_PIECES:\t{player_pieces}')
        # print(f'ENEMY_PIECES:\t{enemy_pieces}')
        # print(f'DICE:\t{dice}')

        for i, piece in enumerate(player_pieces):
            piece_state = self.get_effect_of_move(dice, piece, enemy_pieces)
            # print(f'PIECE {i} POS: {piece}')
            # piece_state.print_piece_next_state()
            self.abstract_state_rep[i][:] = piece_state.piece_state_as_np_array()
        
        return self.abstract_state_rep
    
    def next_move_abstract_state_rep(self, dice, player_pieces, enemy_pieces):
        # TODO: Implement move selection based on network trained with an abstract state representation as input.
        move_pieces = self.get_pieces_that_can_move(dice)
        
        if len(move_pieces) == 1:
            return move_pieces[0]
        
        can_move_piece_list = [0, 0, 0, 0]
        for move_piece_idx in move_pieces:
            can_move_piece_list[move_piece_idx] = 1
        
        next_move_preferences = np.zeros(4)
        next_states = self.generate_abstract_net_input(dice, player_pieces, enemy_pieces)
        for move_piece_idx in range(4):
            if can_move_piece_list[move_piece_idx] != 0:
                #     score = 0
                #     next_move_preferences[].append(0)
                # else:
                net_input = next_states[move_piece_idx]#list(itertools.chain.from_iterable(next_states[move_piece_idx]))
                score = self.net.activate(net_input)
                # print(score)
                next_move_preferences[move_piece_idx] = score[0]
                # possible_moves.append(NEATMove(move_piece_idx, score))
        
        next_move_sorted = np.argsort(next_move_preferences)
        
        next_move = []
        for move in next_move_sorted[::-1]:
            if move in move_pieces:
                next_move = move
                break
        return next_move
        
    def next_move_2(self, dice, board_state):
        move_pieces = self.get_pieces_that_can_move(dice)

        if len(move_pieces) == 1:
            return move_pieces[0]
        
        can_move_piece_list = [0, 0, 0, 0]
        for move_piece_idx in move_pieces:
            can_move_piece_list[move_piece_idx] = 1

        board_state_flattend = list(itertools.chain.from_iterable(board_state[1:5]))
        # print(board_state[1:5])
        # board_state_flattend = list(itertools.chain.from_iterable(board_state))

        # print(np.shape(board_state_flattend))

        net_input = np.concatenate((dice, board_state_flattend), axis=None)
        net_output = self.net.activate(net_input)

        next_move_preferences = np.array(net_output) * np.array(can_move_piece_list)
        next_move_sorted = np.argsort(next_move_preferences)
        
        next_move = []
        for move in next_move_sorted[::-1]:
            if move in move_pieces:
                next_move = move
                break
        
        return next_move


    def next_move(self, dice, _player_pieces, _enemy_pieces):
        move_pieces = self.get_pieces_that_can_move(dice)

        if len(move_pieces) == 1:
            return move_pieces[0]

        can_move_piece_list = [0, 0, 0, 0]
        for move_piece_idx in move_pieces:
            can_move_piece_list[move_piece_idx] = 1

        own_pieces_pos = _player_pieces
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
        # print(type(net_input))
        # print(net_input)
        
        # # Advantage calculations:
        # # TODO: Look at different inputs to the network and see how it affects the agents performance:
        # # # 1) Advantage: sum of tiles betwween all 4 tokens and goal for self and sum of tiles to goal for all 4 tokens for each of the 3 opponents
        # own_tiles_to_goal = sum(57 - own_pieces_pos)
        # # print(own_tiles_to_goal)
        # enemy_tiles_to_goal = [sum(57 - pieces) for pieces in _enemy_pieces]
        # # print(enemy_tiles_to_goal)
        # print(enemy_tiles_to_goal - own_tiles_to_goal)
        # # # 1a) State at t+1 by moving the token will result in a risky state
        # # # 1b) State at t+1 will result in recovering the advantage
        # # # NOTE: advantage is min(enemy_tiles_to_goal, key=lambda x:abs(x-own_tiles_to_goal))
        # print(f'approx. advantage = {max(enemy_tiles_to_goal-own_tiles_to_goal, key=lambda x:abs(x-own_tiles_to_goal))}')
        # # # 2) Pieces' risk of being killed? (propability of each token being killed by an opponent in one turn)
        # # # 2a) Should result in higher values for moving the most delayed piece. If no risk for any tokens, this will no affect the action selection.

        net_output = self.net.activate(net_input)
        # print(f'type(net_output)={type(net_output)}')

        # # Print the input of the net:
        # print("\n\tINPUT:")
        # print(f'\n\tINPUT:\t{net_input}')
        # # # Get the output move from the network and choose the preferred legal move based on net output
        # print("\tOUTPUT:")
        # print(f'\tOUTPUT:\t{np.round(net_output, decimals=5)}')
        # # print(f'\tBEFORE OUTPUT MASK:\n\t{net_output}')
        # # next_move = min(move_pieces, key=lambda x:abs(x-net_output[0]))
        
        next_move_preferences = np.array(net_output) * np.array(can_move_piece_list)
        # print(f'\t\tAFTER OUTPUT MASK:\n\t\t{next_move_preferences}')
        # # print(next_move_preferences)
        next_move_sorted = np.argsort(next_move_preferences)
        
        next_move = []
        for move in next_move_sorted[::-1]:
            if move in move_pieces:
                next_move = move
                break
        # next_move = next_move_sorted[::-1]

        # print(f'\t\tACTION PREFERENCE:\n\t\t{next_move}')
        # print(f'\t\tCHOSEN ACTION:\n\t\t{next_move[0]}')
        # return next_move[0] #min(move_pieces, key=lambda x:abs(x-net_output[0]))
        return next_move

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
    # print("Moves per sec:", moves_per_sec)

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

############### Attemp at init Neat player from genome params - not successfull
# class BestNEATPlayer(Player):
#     def load_neat_net(self, _config):
#         self.genome = neat.DefaultGenome(key=26)
#         # Setup params of best genome:
                
#         # DefaultNodeGene(key=26, bias=0.7055400746108504, response=1.0, activation=sigmoid, aggregation=sum))
#         self.genome.nodes = {'0': neat.genes.DefaultNodeGene(key=0, bias=0.847109424165768, response=1.0, activation='sigmoid', aggregation=sum),
#                              '26': neat.genes.DefaultNodeGene(key=26, bias=0.7055400746108504, response=1.0, activation='sigmoid', aggregation=sum)}
#         self.genome.connections = {neat.genes.DefaultConnectionGene(key=(-8, 0), weight=0.2909629425132466, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-8, 26), weight=1.0600724647637576, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-7, 0), weight=0.6904576012917878, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-7, 26), weight=1.2376852237596465, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-6, 0), weight=-1.8264530205496488, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-6, 26), weight=0.20402443501278095, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-5, 0), weight=-0.15339405692719424, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-5, 26), weight=1.4803382235725915, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-4, 0), weight=0.6215101999583793, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-4, 26), weight=-0.27315269143048665, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-3, 0), weight=0.6633100586245813, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-3, 26), weight=0.9866490432677778, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-2, 0), weight=-0.10201759284315223, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-2, 26), weight=-0.27482991509394106, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-1, 0), weight=1.4222920503544052, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(-1, 26), weight=-0.5433930664667177, enabled=True),
#                             neat.genes.DefaultConnectionGene(key=(26, 0), weight=-0.3112685735210641, enabled=True)
#                             }
        
#         self.genome.fitness = 65.0

#         self.config = _config
#         self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

#         self.abstract_state_rep = np.zeros((4,8))
    
#     def generate_abstract_net_input(self, dice, player_pieces, enemy_pieces):
#         for i, piece in enumerate(player_pieces):
#             piece_state = self.get_effect_of_move(dice, piece, enemy_pieces)
#             self.abstract_state_rep[i][:] = piece_state.piece_state_as_np_array()
#         return self.abstract_state_rep
    
#     def next_move_abstract_state_rep(self, dice, player_pieces, enemy_pieces):
#         move_pieces = self.get_pieces_that_can_move(dice)
        
#         if len(move_pieces) == 1:
#             return move_pieces[0]
        
#         can_move_piece_list = [0, 0, 0, 0]
#         for move_piece_idx in move_pieces:
#             can_move_piece_list[move_piece_idx] = 1
        
#         next_move_preferences = np.zeros(4)
#         next_states = self.generate_abstract_net_input(dice, player_pieces, enemy_pieces)
#         for move_piece_idx in range(4):
#             if can_move_piece_list[move_piece_idx] != 0:
#                 net_input = next_states[move_piece_idx]
#                 score = self.net.activate(net_input)
#                 next_move_preferences[move_piece_idx] = score[0]
#         next_move_sorted = np.argsort(next_move_preferences)
        
#         next_move = []
#         for move in next_move_sorted[::-1]:
#             if move in move_pieces:
#                 next_move = move
#                 break
#         return next_move
    
#     def get_genome(self):
#         return self.genome


# def eval_best_neat_player(config_file):
#     # Load configuration.
#     config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                          config_file)

#     # # Create the population, which is the top-level object for a NEAT run.
#     # p = neat.Population(config)
    
#     # # # Restore population from checkpoint: 
#     # # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')

#     # # Add a stdout reporter to show progress in the terminal.
#     # p.add_reporter(neat.StdOutReporter(True))
#     # stats = neat.StatisticsReporter()
#     # p.add_reporter(stats)
#     # p.add_reporter(neat.Checkpointer(1))

#     # # Run for up to 300 generations.
#     # winner = p.run(eval_genomes, 1)

#     # # Display the winning genome.
#     best_genome = BestNEATPlayer(config)
#     print('\nBest genome:\n{!s}'.format(best_genome.get_genome()))



if __name__ == '__main__':
    # # Determine path to configuration file. This path manipulation is
    # # here so that the script will run successfully regardless of the
    # # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'player_neat_config.txt')
    run_neat(config_path)

