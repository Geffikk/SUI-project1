from typing import List, Union, Tuple, Dict, Optional
import copy
import csv

import numpy

import math

from dicewars.ai.gf.TrainModel import TrainModel

import torch
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.area import Area
from dicewars.client.game.board import Board

GLOBAL = False

class AI:
    __DEPTH = 1
    __MAX_NUMBER_OF_TRANSFERS = 6
    __SUPPORT_FROM_BEHIND = 1

    __MAX_TRANSFERS_FROM_BEHIND = 3
    __MAX_TRANSFERS_BEFORE_BATTLE = 5
    __MIN_TRANSFER_VALUE = 3

    def __write_to_csv(self, dict):
        self.csv_file = open('training_data.csv', 'a')
        fieldnames = ['game_result', 'enemies', 'enemies_areas', 'enemies_dice', 'my_dice', 'my_areas', 'border_areas', 'border_dice', 'regions', 'enemies_regions', 'biggest_region']
        writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        writer.writerow(dict)
        self.csv_file.close()


    def __get_number_of_enemies_area(self, board: Board, player_name):
        number_of_areas = 0
        number_of_dices = 0
        number_of_regions = 0

        for i in range(board.nb_players_alive()):
            if i+1 != player_name:
                number_of_regions += len(board.get_players_regions(i+1))

                areas = board.get_player_areas(i+1)
                number_of_areas += len(areas)

                number_of_dices += board.get_player_dice(i+1)

        return number_of_areas, number_of_dices, number_of_regions

    def __get_dice_on_border(self, board: Board, player_name):
        result = 0
        areas = board.get_player_border(player_name)
        for area in areas:
            result += area.get_dice()

        return result

    def __get_biggest_region(self, board: Board, player_name):
        regions = board.get_players_regions(player_name)

        region_size = 0
        max = 0
        for region in regions:
            for i in region:
                area = board.get_area(i)
                region_size += area.get_dice()

            if region_size > max:
                max = region_size

        return max

    def __init__(self, player_name, board, players_order, max_transfers):
        # self.csv_file = open('training_data.csv', 'a')
        # self.fieldnames = ['game_result', 'enemies', 'enemies_areas', 'enemies_dice', 'my_dice', 'my_areas', 'border_areas', 'border_dice', 'regions', 'enemies_regions',
        #              'biggest_region']
        # writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        # writer.writeheader()

        # sniffer = csv.Sniffer()
        # sample_bytes = 32
        #print(sniffer.has_header(open("training_data.csv").read(sample_bytes)))
        # self.csv_file.close()

        self.players_order = players_order
        self.model = TrainModel.load_model()

        self.player_name = player_name
        self.sum_of_dices = 0
        self.promising_attack = None
        self.attack_depth_two = None
        self.middle_areas = []
        self.areas_distance_from_border = {}
        self.prob_of_successful_attack = 0

        self.agresivity_index = 1
        self.area_win_lose = -1
        self.number_areas_previous = 1
        self.training_data = {}

        self.players_order=players_order
        #potreba mit nasi ai v seznamu na prvnim miste
        while player_name != self.players_order[0]:
            self.players_order.append(self.players_order.pop(0))

        if board.nb_players_alive() == 2:
            self.treshold = 0.2
            self.score_weight = 3
        else:
            self.treshold = 0.4
            self.score_weight = 2

    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        num_of_areas, num_of_dice, number_of_regions = self.__get_number_of_enemies_area(board, self.player_name)
        # test_data = {}
        # test_data['enemies'] = board.nb_players_alive()
        # test_data['enemies_areas'] = num_of_areas
        # test_data['enemies_dice'] = num_of_dice
        # test_data['my_dice'] = board.get_player_dice(self.player_name)
        # test_data['my_areas'] = len(board.get_player_areas(self.player_name))
        # test_data['border_areas'] = len(board.get_player_border(self.player_name))
        # test_data['border_dice'] = self.__get_dice_on_border(board, self.player_name)
        # test_data['regions'] = len(board.get_players_regions(self.player_name))
        # test_data['enemies_regions'] = number_of_regions
        # test_data['biggest_region'] = self.__get_biggest_region(board, self.player_name)
        # test_data['game_result'] = -1
        # self.__write_to_csv(test_data)
        #print("Start geffik AI Player-" + str(self.player_name) + " turn")
        self.promising_attack = []
        self.sum_of_dices = 0
        self.middle_areas = [] # todo
        self.prob_of_successful_attack = 0

        self.nb_transfers_this_turn = nb_transfers_this_turn

        """Try to use half of move commands to transfer units from behind"""
        transfer_command = self.transfer_from_behind_decider(nb_transfers_this_turn, board)

        if transfer_command is not None:
            return transfer_command  # Return TransferCommand()

        """Using the rest of commands to support borders"""
        transfer_command = self.transfer_on_border_decider(nb_transfers_this_turn, board)

        if transfer_command is not None:
            return transfer_command  # Return TransferCommand()

        """If there are any moves left try to move more units from behind"""
        transfer_command = self.find_areas_for_transfer(nb_transfers_this_turn, board)

        if transfer_command is not None:
            return transfer_command  # Return TransferCommand()

        if self.number_areas_previous < len(board.get_player_areas(self.player_name)):
            self.area_win_lose = 1
        else:
            self.area_win_lose = -1
        self.number_areas_previous = len(board.get_player_areas(self.player_name))

        attack_command = self.attack_decider(time_left, board)

        if attack_command is not None:
            return attack_command  # Return AttackCommand()

        """Try to make more moves after attacking"""
        """Using the rest of commands to support borders"""
        transfer_command = self.transfer_on_border_decider(nb_transfers_this_turn, board)

        if transfer_command is not None:
            return transfer_command  # Return TransferCommand()

        """If there are any moves left try to move more units from behind"""
        transfer_command = self.find_areas_for_transfer(nb_transfers_this_turn, board)

        if transfer_command is not None:
            return transfer_command  # Return TransferCommand()

        return EndTurnCommand()

    def attack_decider(self, time_left, board):
        self.action_attack(board, time_left, self.__DEPTH)

        if len(self.promising_attack) == 0:
            return EndTurnCommand()

        if self.prob_of_successful_attack != 0:
            can_attack = self.may_attack(self.prob_of_successful_attack, time_left, board)
            if can_attack is False:
                return EndTurnCommand()

        source = self.promising_attack[0]
        target = self.promising_attack[1]

        if source is not None or target is not None:
            return BattleCommand(source.get_name(), target.get_name())
        else:
            return EndTurnCommand()

    def transfer_on_border_decider(self, nb_transfers_this_turn, board):
        if nb_transfers_this_turn < self.__MAX_NUMBER_OF_TRANSFERS:
            vulnerable_area, neighbour_area = self.transfer_dice_to_border(board)
            if vulnerable_area is not None and neighbour_area is not None:
                return TransferCommand(neighbour_area.get_name(), vulnerable_area.get_name())
        return None

    def transfer_from_behind_decider(self, nb_transfers_this_turn, board):
        """USE HALF OF MOVE COMMANDS TO TRANSFER UNITS FROM BEHIND"""
        if nb_transfers_this_turn < self.__MAX_TRANSFERS_FROM_BEHIND:
            return self.find_areas_for_transfer(nb_transfers_this_turn, board)
        return None

    def find_areas_for_transfer(self, nb_transfers_this_turn, board):
        if nb_transfers_this_turn != self.__MAX_NUMBER_OF_TRANSFERS:
            full_area, vulnerable_area = self.move_dice_from_behind_to_front(board)

            if full_area is not None and vulnerable_area is not None:
                return TransferCommand(full_area.get_name(), vulnerable_area.get_name())

        return None
        
    def eval_node(self, board: Board, player_name):
        """
        Description: heuristic for board evaluation based on machine learning.

        Parameters:
            board: Game board.
            player_name: name of player from whose perspective the board is judged

        Return: model, numerical evaluation of board
        """

        num_of_areas, num_of_dice, number_of_regions = self.__get_number_of_enemies_area(board, player_name)
        self.training_data['enemies'] = board.nb_players_alive()
        self.training_data['enemies_areas'] = num_of_areas
        self.training_data['enemies_dice'] = num_of_dice
        self.training_data['my_dice'] = board.get_player_dice(player_name)
        self.training_data['my_areas'] = len(board.get_player_areas(player_name))
        self.training_data['border_areas'] = len(board.get_player_border(player_name))
        self.training_data['border_dice'] = self.__get_dice_on_border(board, player_name)
        self.training_data['regions'] = len(board.get_players_regions(player_name))
        self.training_data['enemies_regions'] = number_of_regions
        self.training_data['biggest_region'] = self.__get_biggest_region(board, player_name)
        vector = []

        for column in self.training_data.values():
            vector.append(column)

        # Aby sa uz neupravovali vahy modelu
        with torch.no_grad():
            # Pre hodnotu 0/1 ako predpovede ci prehra aelob vyhra z daneho stavu
            # pst = TrainModel.threshold(self.model(torch.Tensor([vector])))
            # Teraz mame pravdepodobnost v rozmedzi 0 az 1 (napr. 0.821)
            pst = self.model(torch.Tensor([vector]))

        model=(pst.item()*0.7)
        return model

    def expand_moves(self, board: Board, player_name):
        """
        Description: explosion of node

        Parameters:
            board: Game board.
            player_name: name of player from whose perspective the board is judged

        Return: list of boards that are possible to get by player attacking
        """
        attacks = list(possible_attacks(board, player_name))
        if attacks == []:
            return [board]
        list_of_boards=[]
        for attack in attacks:
            source_area = attack[0]
            target_area = attack[1]                
            prob_of_successful_attack = 0
            if source_area.get_dice() > 1:
                prob_of_successful_attack = probability_of_successful_attack(board, source_area.get_name(), target_area.get_name())
                
            if prob_of_successful_attack < 0.25:
                continue
            atk_power = source_area.get_dice()
            hold_prob = prob_of_successful_attack * probability_of_holding_area(board, target_area.get_name(), atk_power - 1, player_name)
            board_after_simulation = self.simulate_turn(board, source_area, target_area)
            list_of_boards.append(board_after_simulation)
        return list_of_boards
        
    def simulate_game(self,board: Board, player_name, depth):
        """
        Description: state space search

        Parameters:
            board: Game board.
            player_name: name of player from whose perspective the game is judged
            depth: level of submersion

        Return: tmodel, numerical evaluation of state
        """
        tmodel=0
        boards=[board]
        #this player is playing, expanding nodes
        for x in range(depth):
            boards_inner=[]
            for b in boards:
                boards_inner=boards_inner+(self.expand_moves(b,player_name))                
            boards=boards_inner
        #other players are playing
        for player in self.players_order[1:]:
            for b in boards:
                boards_help=[b]
                boards_inner_help=[]
                rememberb=b
                for x in range(depth):
                    for b2 in boards_help:
                        boards_inner_help=boards_inner_help+(self.expand_moves(b2,player))
                    boards_help=boards_inner_help
                #now we have possible expansions for node b after -depth- turns of player
                for b3 in boards_help:
                    player_model=0
                    model=self.eval_node(b3, player)
                    #we suppose they will chose route which is best for them
                    if model>tmodel:
                        playermodel=model
                        rememberb=b3
                #therefore the node from our list will change to node played by player
                boards.remove(b)
                boards.append(rememberb)
        #now we have responses of every player to every node
        for b in boards:
            model=self.eval_node(b, player_name)
            if model>tmodel:
                tmodel=model
        return tmodel

                    

    def action_attack(self, board: Board, time_left, depth):
        """
           Description: Find most useful attack.

           Parameters:
               board: Game board.
               time_left: time_left: Time remaining.
               depth: Depth of recursion.

           Return: Return, board after simulation.
        """
        attacks = list(possible_attacks(board, self.player_name))
        number_of_dices = board.get_player_dice(self.player_name)

        if depth == 0 or attacks == []:
            return number_of_dices

        for attack in attacks:
            if depth == self.__DEPTH:
                self.attack_depth_two = attack

            source_area = attack[0]
            target_area = attack[1]

            prob_of_successful_attack = 0
            if source_area.get_dice() > 1:
                prob_of_successful_attack = probability_of_successful_attack(board, source_area.get_name(), target_area.get_name())

            if prob_of_successful_attack < 0.25:
                continue

            atk_power = source_area.get_dice()
            hold_prob = prob_of_successful_attack * probability_of_holding_area(board, target_area.get_name(), atk_power - 1, self.player_name)

            board_after_simulation = copy.deepcopy(board)
            board_after_simulation = self.simulate_turn(board_after_simulation, source_area, target_area)

            number_of_dices = self.action_attack(board_after_simulation, time_left, depth - 1)
            atk_power = float(atk_power / 8)
            if time_left>10:
                model=self.simulate_game(board_after_simulation, self.player_name, 4)
                #print("reg4:" + str(time_left))
            elif time_left>5:
                model=self.simulate_game(board_after_simulation, self.player_name, 3)
                #print("reg3:" + str(time_left))
            else:
                model=self.simulate_game(board_after_simulation, self.player_name, 2)
                #print("reg2" + str(time_left))
            #print("model gives value:" + str(model))
            median = (model + (prob_of_successful_attack*0.6) + (hold_prob*1.4) + (atk_power*1.3)) / 4
            if median > self.prob_of_successful_attack and depth == self.__DEPTH:
                self.prob_of_successful_attack = median
                self.promising_attack = attack
                #print("current highest value:" + str(model))

            # TODO.md - Tento trash code treba nahradiť normalnym prehladavanim stavoveho priestoru
            #if median > self.prob_of_successful_attack and depth == self.__DEPTH:
            #    #self.sum_of_dices = number_of_dices
            #    self.prob_of_successful_attack = median
            #    self.promising_attack = attack
        #print("attack selected")
        return number_of_dices

    @staticmethod
    def simulate_turn(board, source, target):
        """
        Description: One turn simulation.

        Parameters:
            board: Game board.
            source: Source area.
            target: Target area.

        Return: Return, board after simulation
        """
        src_name = source.get_name()
        src_tmp = board.get_area(src_name)

        tgt_name = target.get_name()
        tgt_tmp = board.get_area(tgt_name)

        tgt_tmp.set_owner(source.get_owner_name())
        tgt_tmp.set_dice(source.get_dice() - 1)

        src_tmp.set_dice(1)

        return board

    def transfer_dice_to_border(self, board: Board) -> Tuple[Area, Area]:
        """
        Description: Transfer dice on border. Searches neighbours of
                    each border area and supports area which will has
                    the biggest difference of probability of hold before support and after support

        Parameters:
            board: Game board.

        Return: Return, area which can transfer her dice on border
        """
        borders = board.get_player_border(self.player_name)

        target_border = None
        max_prob_dif = [None, -1]

        for border_area in borders:
            if border_area.get_dice() != 8:
                border_area_neighbours = border_area.get_adjacent_areas_names()
                curr_hold_prob = probability_of_holding_area(board,border_area.name,border_area.get_dice(),self.player_name)
                for area_name in border_area_neighbours:
                    area = board.get_area(area_name)
                    if area.get_owner_name() == self.player_name and area not in borders:
                        new_dice = area.get_dice() + border_area.get_dice()
                        if new_dice > 8:
                            new_dice = 8
                        supp_hold_prob = probability_of_holding_area(board, border_area.name, new_dice,self.player_name)
                        curr_prob_diff = math.fabs(supp_hold_prob-curr_hold_prob)

                        if curr_prob_diff > max_prob_dif[1]:
                            max_prob_dif[0] = area
                            max_prob_dif[1] = curr_prob_diff
                            target_border = border_area

        return target_border, max_prob_dif[0]

    def get_vulnerable_area(self, board: Board) -> Optional[Area]:
        """
        Description: Get vulnerable area

        Parameters:
            board: Game board.

        Return: Return area with minimum probability of retention
        """
        vulnerable_areas = board.get_player_border(self.player_name)
        result_area = None

        for i, area in enumerate(vulnerable_areas):
            if area.get_dice() <= 6:
                del vulnerable_areas[i]

        min_prob_hold = 1.0
        for vulnerable_area in vulnerable_areas:
            prob_of_old_area = probability_of_holding_area(board, vulnerable_area.get_name(), vulnerable_area.get_dice(), self.player_name)
            if prob_of_old_area < min_prob_hold:
                result_area = vulnerable_area
                min_prob_hold = prob_of_old_area

        if result_area is None:
            return None

        return result_area

    def get_neighbours_of_vulnerable_area(self, board: Board, vulnerable_area: Area) -> Optional[Area]:
        """
        Description: Get not on border neighbour area

        Parameters:
            board: Game board.
            vulnerable_area: Vulnerable area

        Return: Return neighbour area which can support vulnerable area
        """
        neighbours = vulnerable_area.get_adjacent_areas_names()

        for adj in neighbours:
            adjacent_area = board.get_area(adj)

            is_own_area = adjacent_area.get_owner_name() == self.player_name
            is_area_at_border = board.is_at_border(adjacent_area)

            if is_own_area and is_area_at_border is False and adjacent_area.get_dice() > 2:
                vulnerable_area.set_dice(vulnerable_area.get_dice() + adjacent_area.get_dice() - 1)

                return adjacent_area

            elif is_own_area and is_area_at_border is False:
                if adjacent_area not in self.middle_areas:
                    self.middle_areas.append(adjacent_area)

        return None

    def move_dice_from_behind_to_front(self, board: Board):
        """
        Description: Move dice from behind closer to border.

        Parameters:
            board: Game board.

        Return: Supporting area and area which need to be supported
        """
        areas_distance_from_border = {}
        areas_distance_from_border = self.fill_area_distance(board, areas_distance_from_border, 0)

        supporting_area = None
        adjacent_area = None
        max_iterations = 7
        banned_areas = []

        while ((supporting_area is None) or (adjacent_area is None)) and (max_iterations != 0):
            if supporting_area is not None and adjacent_area is None:
                banned_areas.append(supporting_area)

            area_level, supporting_area = self.get_area_from_behind(areas_distance_from_border, banned_areas)
            adjacent_area = self.find_area_which_need_support(supporting_area, areas_distance_from_border, area_level, board)
            max_iterations -= 1

        if supporting_area is None or adjacent_area is None:
            return None, None
        elif supporting_area is not None and adjacent_area is not  None:
            transfer_value = self.calculate_transfer_value(supporting_area, adjacent_area)
            if transfer_value < self.__MIN_TRANSFER_VALUE:
                supporting_area,adjacent_area = self.propagate_support(supporting_area,adjacent_area, board, areas_distance_from_border, area_level)
        return supporting_area, adjacent_area

    @staticmethod
    def get_area_from_behind(areas_distance_from_border, banned_areas):
        """
        Descriptions: Get one area from back rows.

        Parameters:
            areas_distance_from_border: Dictionary which contain all areas ordered by distance from border (from nearest).
            banned_areas: List of areas which was researched

        Return: Area which has, more then X dice.
        """
        if areas_distance_from_border is None:
            return -1, None
        val_max = 0
        max_dist = -1
        max_area = None
        for i in range(2, len(areas_distance_from_border)):
            for a in areas_distance_from_border[i]:
                if a not in banned_areas and a.get_dice() > 1:
                    value = 0.75 * i * 1.5 * a.get_dice()
                    if value > val_max:
                        max_area = a
                        val_max = value
                        max_dist = i
        return max_dist, max_area

    def propagate_support(self, starting_area,adjacent_area,board,areas_distance_from_border,area_level):
        open_list = [(starting_area,adjacent_area)]
        closed_list = []
        possible_ends = []
        while len(open_list) != 0:
            curr_pair = open_list.pop()
            curr_area = curr_pair[1]
            curr_distance = self.get_distance_from_border(areas_distance_from_border,curr_area)
            for neighbour_name in curr_area.get_adjacent_areas_names():
                neighbour_area = board.get_area(neighbour_name)
                neighbour_distance = self.get_distance_from_border(areas_distance_from_border,neighbour_area)
                neighbour_transfer_value = self.calculate_transfer_value(curr_area,neighbour_area)
                generated_child = False
                if neighbour_transfer_value < self.__MIN_TRANSFER_VALUE and neighbour_distance < curr_distance and neighbour_distance != 0 and neighbour_area not in closed_list:
                    generated_child = True
                    open_list.append((curr_area,neighbour_area))
            if not generated_child and curr_area != adjacent_area and curr_area not in possible_ends:
                possible_ends.append(curr_pair)
            closed_list.append(curr_area)

        max_val = 0
        chosen_support_end = None
        chosen_adj_end = None
        if len(possible_ends) > 0:
            for poss_end in possible_ends:
                poss_end_area = poss_end[1]
                if poss_end_area.get_dice() != 8:
                    transfer_val = poss_end_area.get_dice() / self.get_distance_from_border(areas_distance_from_border,poss_end_area)
                    if transfer_val > max_val:
                        max_val = transfer_val
                        chosen_support_end = poss_end[0]
                        chosen_adj_end = poss_end_area
        if chosen_support_end is None or chosen_adj_end is None or (chosen_adj_end.get_dice() == 8):
            chosen_support_end = starting_area
            chosen_adj_end = adjacent_area

        return chosen_support_end, chosen_adj_end

    @staticmethod
    def get_distance_from_border(areas_distance_from_border,area):
        val_list = list(areas_distance_from_border.values())
        position = -1
        for area_list in val_list:
            if area in area_list:
                return val_list.index(area_list)
        return position

    @staticmethod
    def calculate_transfer_value(transfer_area, receive_area):
        able_to_transfer_val = transfer_area.get_dice() - 1
        able_to_receive_val = 8 - receive_area.get_dice()

        transfer_value = able_to_transfer_val
        if able_to_receive_val < able_to_transfer_val:
            transfer_value = able_to_receive_val
        return transfer_value

    def find_area_which_need_support(self, supporting_area, areas_distance_from_border, area_level, board):
        """
        Description: Find area which need to be supported

        Parameters:
            supporting_area: -
            areas_distance_from_border: Dictionary which contain all areas ordered by distance from border (from nearest).
            area_level: Indicator which indicates how far i searching for from border.
            board: Actual bord.

        Return: Area which which need, to be supported
        """
        weak_area_val = 10000
        supp_area = None

        if supporting_area is None or areas_distance_from_border is None:
            return None, None

        adj_areas = supporting_area.get_adjacent_areas_names()
        for adj in adj_areas:
            adj_area_distance = self.get_distance_from_border(areas_distance_from_border,board.get_area(adj))

            if adj_area_distance < area_level:
                cur_val = 1.2*adj_area_distance + board.get_area(adj).get_dice()
                if cur_val < weak_area_val:
                    weak_area_val = cur_val
                    supp_area = board.get_area(adj)
        return supp_area

    def fill_area_distance(self, board: Board, areas_distance_from_border: Dict[int, List[Area]], depth):
        """
        Description: Fill dictionary with areas ordered by distance.

        Parameters:
            board: Actual board
            areas_distance_from_border: Dictionary which contain all areas ordered by distance from border (from nearest).
            depth: depth of recursion

        Return: areas_distance_from_border.
        """
        if len(areas_distance_from_border) == 0:
            areas_distance_from_border = {0: board.get_player_border(self.player_name)}

        player_border_arr = areas_distance_from_border[depth]
        new_player_border_arr = []

        if depth == 6 or player_border_arr == []:
            return areas_distance_from_border

        for player_area in player_border_arr:
            adj_areas = player_area.get_adjacent_areas_names()

            for adj in adj_areas:
                neighbour_area = board.get_area(adj)

                if neighbour_area.get_owner_name() == self.player_name and neighbour_area not in new_player_border_arr:
                    is_there = False

                    for i in range(len(areas_distance_from_border)):
                        if neighbour_area in areas_distance_from_border[i]:
                            is_there = True

                    if is_there is False:
                        new_player_border_arr.append(neighbour_area)

        areas_distance_from_border[depth + 1] = new_player_border_arr
        return self.fill_area_distance(board, areas_distance_from_border, depth + 1)

    def may_attack(self, win_chance, time_left, board):
        """
        Description: Calculate win probability connected with heuristic like (agresivity index)

        Parameters:
            win_chance: Probability for take over enemy area
            time_left: Time remaining
            board: Game board

        Return: True if attack is promising, False otherwise
        """
        self.agresivity_index = (self.agresivity_index - ((win_chance / (board.nb_players_alive() * time_left)) * self.area_win_lose) * 1.7)

        if self.agresivity_index > 1:
            self.agresivity_index = 1
        elif self.agresivity_index < 0:
            self.agresivity_index = 0

        if board.nb_players_alive() >= 4:
            tresh_hold = 0.48 #* self.agresivity_index
        elif board.nb_players_alive() == 3:
            tresh_hold = 0.42 #* self.agresivity_index
        else:
            tresh_hold = 0.33 #* self.agresivity_index

        if win_chance > tresh_hold:
            return True
        else:
            return False

    def get_largest_region(self):
        """Get size of the largest region, including the areas within

        Attributes
        ----------
        largest_region : list of int
            Names of areas in the largest region

        Returns
        -------
        int
            Number of areas in the largest region
        """
        self.largest_region = []

        players_regions = self.board.get_players_regions(self.player_name)
        max_region_size = max(len(region) for region in players_regions)
        max_sized_regions = [region for region in players_regions if len(region) == max_region_size]

        self.largest_region = max_sized_regions[0]
        return max_region_size
