from typing import List, Union, Tuple, Dict, Optional
import copy
import csv

import numpy

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

    def __write_to_csv(self, dict):
        self.csv_file = open('training_data.csv', 'a')
        fieldnames = ['game_result', 'enemies', 'enemies_areas', 'enemies_dice', 'my_dice', 'my_areas', 'border_areas', 'border_dice', 'regions', 'enemies_regions', 'biggest_region']
        writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        writer.writerow(dict)
        self.csv_file.close()


    def __get_number_of_enemies_area(self, board: Board):
        number_of_areas = 0
        number_of_dices = 0
        number_of_regions = 0

        for i in range(board.nb_players_alive()):
            if i+1 != self.player_name:
                number_of_regions += len(board.get_players_regions(i+1))

                areas = board.get_player_areas(i+1)
                number_of_areas += len(areas)

                number_of_dices += board.get_player_dice(i+1)

        return number_of_areas, number_of_dices, number_of_regions

    def __get_dice_on_border(self, board: Board):
        result = 0
        areas = board.get_player_border(self.player_name)
        for area in areas:
            result += area.get_dice()

        return result

    def __get_biggest_region(self, board: Board):
        regions = board.get_players_regions(self.player_name)

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
        #self.csv_file = open('training_data.csv', 'a')
        #fieldnames = ['game_result', 'enemies', 'enemies_areas', 'enemies_dice', 'my_dice', 'my_areas', 'border_areas', 'border_dice', 'regions', 'enemies_regions',
        #              'biggest_region']
        #writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        #writer.writeheader()
        #self.csv_file.close()

        #sniffer = csv.Sniffer()
        #sample_bytes = 32
        #print (sniffer.has_header(
        #    open("training_data.csv").read(sample_bytes)))
        #self.csv_file.close()
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

        if board.nb_players_alive() == 2:
            self.treshold = 0.2
            self.score_weight = 3
        else:
            self.treshold = 0.4
            self.score_weight = 2

    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        #print("Start geffik AI Player-" + str(self.player_name) + " turn")
        self.promising_attack = []
        self.sum_of_dices = 0
        self.middle_areas = [] # todo
        self.prob_of_successful_attack = 0

        transfer_command = self.transfer_from_behind_decider(nb_transfers_this_turn, board)

        #print(nb_transfers_this_turn)

        if transfer_command is not None:
            return transfer_command  # Return TransferCommand()

        transfer_command = self.transfer_on_border_decider(nb_transfers_this_turn, board)

        if transfer_command is not None:
            return transfer_command  # Return TransferCommand()

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
            while nb_transfers_this_turn < self.__MAX_NUMBER_OF_TRANSFERS:
                for i in range(len(board.get_player_border(self.player_name))):
                    vulnerable_area, neighbour_area = self.transfer_dice_to_border(board)
                    if vulnerable_area is not None and neighbour_area is not None:
                        return TransferCommand(neighbour_area.get_name(), vulnerable_area.get_name())
                    else:
                        pass

                if len(self.middle_areas) != 0:
                    vulnerable_area, neighbour_area = self.support_middle_areas(board, self.middle_areas, 5)
                    if vulnerable_area is not None and neighbour_area is not None:
                        return TransferCommand(neighbour_area.get_name(), vulnerable_area.get_name())
                    else:
                        break
                else:
                    break

        return None

    def transfer_from_behind_decider(self, nb_transfers_this_turn, board):
        num_of_areas = 3

        if nb_transfers_this_turn < num_of_areas:
            return self.find_areas_for_transfer(nb_transfers_this_turn, board)

        return None

    def find_areas_for_transfer(self, nb_transfers_this_turn, board):
        if nb_transfers_this_turn != self.__MAX_NUMBER_OF_TRANSFERS:
            full_area, vulnerable_area = self.move_dice_from_behind_to_front(board)

            if full_area is not None and vulnerable_area is not None:
                return TransferCommand(full_area.get_name(), vulnerable_area.get_name())

        return None

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

            num_of_areas, num_of_dice, number_of_regions = self.__get_number_of_enemies_area(board_after_simulation)
            self.training_data['enemies'] = board_after_simulation.nb_players_alive()
            self.training_data['enemies_areas'] = num_of_areas
            self.training_data['enemies_dice'] = num_of_dice
            self.training_data['my_dice'] = board_after_simulation.get_player_dice(self.player_name)
            self.training_data['my_areas'] = len(board_after_simulation.get_player_areas(self.player_name))
            self.training_data['border_areas'] = len(board_after_simulation.get_player_border(self.player_name))
            self.training_data['border_dice'] = self.__get_dice_on_border(board_after_simulation)
            self.training_data['regions'] = len(board_after_simulation.get_players_regions(self.player_name))
            self.training_data['enemies_regions'] = number_of_regions
            self.training_data['biggest_region'] = self.__get_biggest_region(board_after_simulation)
            vector = []

            for column in self.training_data.values():
                vector.append(column)

            pst = self.model(torch.Tensor([vector]))
            atk_power = float(atk_power / 8)
            median = ((pst.item()*0.8) + (prob_of_successful_attack*0.8) + (hold_prob*1.2) + (atk_power*1.2)) / 4

            if median > self.prob_of_successful_attack and depth == self.__DEPTH:
                self.prob_of_successful_attack = median
                self.promising_attack = attack

            # TODO - Tento trash code treba nahradiť normalnym prehladavanim stavoveho priestoru
            #if median > self.prob_of_successful_attack and depth == self.__DEPTH:
            #    #self.sum_of_dices = number_of_dices
            #    self.prob_of_successful_attack = median
            #    self.promising_attack = attack

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
        Description: Transfer dice on border.

        Parameters:
            board: Game board.

        Return: Return, area which can transfer her dice on border
        """
        vulnerable_area = self.get_vulnerable_area(board)
        if vulnerable_area is not None:
            neighbour_area = self.get_neighbours_of_vulnerable_area(board, vulnerable_area)
        else:
            neighbour_area = None

        return vulnerable_area, neighbour_area

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

    def support_middle_areas(self, board: Board, middle_areas: List[Area], depth) -> Union[tuple[None, None], tuple[Area, Area]]:
        """
        Description: Support middle areas

        Parameters:
            board: Game board.
            middle_areas: Areas which are not on border.
            depth: depth of recursion

        Return: Supporting area and area which need to be supported
        """
        new_middle_areas = copy.deepcopy(self.middle_areas)

        for area in middle_areas:
            adjacent_areas = area.get_adjacent_areas_names()

            for adj in adjacent_areas:
                adjacent_area = board.get_area(adj)

                is_owner_name = adjacent_area.get_owner_name() == self.player_name
                is_at_border = board.is_at_border(adjacent_area) is False
                is_in_middle_areas = adjacent_area not in new_middle_areas
                has_dice = adjacent_area.get_dice() > 3

                if is_owner_name and is_at_border is False and has_dice and is_in_middle_areas is False:
                    return area, adjacent_area

                elif is_owner_name and is_at_border is False and is_in_middle_areas is False:
                    if adjacent_area not in new_middle_areas:
                        new_middle_areas.append(adjacent_area)

        if len(new_middle_areas) == 0 or depth == 0:
            adjacent_area = None
            area = None
            return area, adjacent_area

        area, adjacent_area = self.support_middle_areas(board, new_middle_areas, depth - 1)
        return area, adjacent_area

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

        while ((supporting_area is None) or (adjacent_area is None)) and (max_iterations == 0):
            if supporting_area is not None and adjacent_area is None:
                banned_areas.append(supporting_area)

            area_level, supporting_area = self.get_area_from_behind(areas_distance_from_border, banned_areas)
            adjacent_area = self.find_area_which_need_support(supporting_area, areas_distance_from_border, area_level, board)
            max_iterations -= 1

        if supporting_area is None or adjacent_area is None:
            return None, None

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
        area_level = -1
        supporting_area = None
        max_val = 0

        if areas_distance_from_border is None:
            return area_level, None

        for i in range(3, len(areas_distance_from_border)):
            for area in areas_distance_from_border[i]:
                if (area.get_dice() >= 5) and (area.get_dice() > max_val) and (area not in banned_areas):
                    area_level = i
                    max_val = area.get_dice()
                    supporting_area = area

        return area_level, supporting_area

    @staticmethod
    def find_area_which_need_support(supporting_area, areas_distance_from_border, area_level, board):
        """
        Description: Find area which need to be supported

        Parameters:
            supporting_area: -
            areas_distance_from_border: Dictionary which contain all areas ordered by distance from border (from nearest).
            area_level: Indicator which indicates how far i searching for from border.
            board: Actual bord.

        Return: Area which which need, to be supported
        """
        weak_area = None
        min_val = 8

        if supporting_area is None or areas_distance_from_border is None:
            return None, None

        adj_areas = supporting_area.get_adjacent_areas_names()
        for adj in adj_areas:
            adjacent_area = board.get_area(adj)

            if (adjacent_area in areas_distance_from_border[area_level - 1]) and (adjacent_area.get_dice() <= min_val):
                min_val = adjacent_area.get_dice()
                weak_area = adjacent_area

        return weak_area

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

        if board.nb_players_alive() == 4:
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