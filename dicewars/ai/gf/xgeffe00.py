import random
import logging
from typing import List, Union, Tuple, Deque, Dict, Optional, Any
import copy
from colorama import Fore

from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.area import Area
from dicewars.client.game.board import Board

class AI:
    __DEPTH = 2
    __MAX_NUMBER_OF_TRANSFERS = 6

    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.cache = []
        self.sum_of_dices = 0
        self.promising_attack = None
        self.attack_depth_two = None
        self.middle_areas = []
        self.banned_areas = []

    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        print("------------------------------------ NEW TURN -------------------------------------------------")
        print("Start geffik AI Player-" + str(self.player_name) + " turn")
        self.promising_attack = []
        self.sum_of_dices = 0
        self.middle_areas = [] # todo
        self.banned_areas = [] # todo

        if nb_transfers_this_turn < self.__MAX_NUMBER_OF_TRANSFERS:
            while nb_transfers_this_turn < self.__MAX_NUMBER_OF_TRANSFERS:
                for i in range(len(board.get_player_border(self.player_name))):
                    print("AI: I trying to support border areas")
                    vulnerable_area, neighbour_area = self.transfer_dice_to_border(board)
                    if vulnerable_area is not None and neighbour_area is not None:
                        print("AI: I supporting area: " + str(vulnerable_area.get_name()) + " from area: " + str(neighbour_area.get_name()))
                        return TransferCommand(neighbour_area.get_name(), vulnerable_area.get_name())
                    else:
                        print("AI: Border area not supported")

                if len(self.middle_areas) != 0:
                    print("AI: I trying to support middle areas")
                    vulnerable_area, neighbour_area = self.support_middle_areas(board, self.middle_areas)
                    if vulnerable_area is not None and neighbour_area is not None:
                        print("AI: I supporting middle area: " + str(vulnerable_area.get_name()) + " from area: " + str(neighbour_area.get_name()))
                        return TransferCommand(neighbour_area.get_name(), vulnerable_area.get_name())
                    else:
                        print("AI: I dont have troops for transfer")
                        break
                else:
                    break


        self.actions_calculation(board, self.__DEPTH)

        if len(self.promising_attack) == 0:
            print("No more possible turns.")
            return EndTurnCommand()

        source = self.promising_attack[0]
        target = self.promising_attack[1]

        if source is not None or target is not None:
            print("Attack!")
            return BattleCommand(source.get_name(), target.get_name())
        else:
            print("No more possible turns.")
            return EndTurnCommand()

    # Attack on most probable
    def actions_calculation(self, board: Board, depth):
        attacks = list(possible_attacks(board, self.player_name))
        print("Num of possible attacks: " + str(len(attacks)) + " in depth: " + str(depth))
        number_of_dices = self.get_player_number_of_dice(board)

        if depth == 0 or attacks == []:
            print("Return from recursion with number of dices: " + str(number_of_dices))
            return number_of_dices

        for attack in attacks:
            if depth == self.__DEPTH:
                print("Storing a possible attack: " + str(attack))
                self.attack_depth_two = attack

            source = attack[0]
            target = attack[1]

            tmp = 0
            if source.get_dice() > 1:
                tmp = probability_of_successful_attack(board, source.get_name(), target.get_name())
            print("My probability on successfull attack is: " + str(tmp))

            if tmp < 0.30:
                print("AI: Probability is too small, i will search for better options")
                continue

            board_after_simulation = copy.deepcopy(board)
            board_after_simulation = self.simulate_turn(board_after_simulation, source, target)

            print("AI: Probability is ok, i will simulate attack and then search deeper in depth: " + str(depth - 1))
            number_of_dices = self.actions_calculation(board_after_simulation, depth - 1)
            print("AI: Back in depth: " + str(depth))

            if number_of_dices >= self.sum_of_dices:
                print("AI: I found solution with more dices, then previous: " +
                      str(self.sum_of_dices) + " actual: " + str(number_of_dices))
                self.sum_of_dices = number_of_dices
                if depth == self.__DEPTH:
                    self.promising_attack = attack
                    print("I found better option, so i replaced current, storing attack source: " +
                          str(self.promising_attack[0].get_name()) + " target: " +
                          str(self.promising_attack[1].get_name()))
            print("######################################################################################")

        print("I already research all my possible attacks in depth: " + str(depth))
        print("Return from recursion with number of dices: " + str(number_of_dices))
        return number_of_dices

    def get_player_number_of_dice(self, board):
        num_of_dice = board.get_player_dice(self.player_name)
        return num_of_dice

    def simulate_turn(self, board, source, target):
        src_name = source.get_name()
        src_tmp = board.get_area(src_name)

        tgt_name = target.get_name()
        tgt_tmp = board.get_area(tgt_name)

        tgt_tmp.set_owner(source.get_owner_name())
        tgt_tmp.set_dice(source.get_dice() - 1)

        src_tmp.set_dice(1)

        return board

    def transfer_dice_to_border(self, board: Board) -> Tuple[Area, Area]:
        vulnerable_area = self.get_vulnerable_area(board)
        if vulnerable_area is not None:
            neighbour_area = self.get_neighbours_of_vulnerable_area(board, vulnerable_area)
        else:
            neighbour_area = None
        return vulnerable_area, neighbour_area

    def get_vulnerable_area(self, board: Board) -> Optional[Area]:
        areas = board.get_player_border(self.player_name)
        vulnerable_areas = []
        result_area = None

        for area in areas:
            if area.get_dice() <= 6:
                vulnerable_areas.append(area)

        min_prob_hold = 1.0
        for vulnerable_area in vulnerable_areas:
            prob_of_old_area = probability_of_holding_area(board, vulnerable_area.get_name(), vulnerable_area.get_dice(), self.player_name)
            if prob_of_old_area < min_prob_hold and vulnerable_area not in self.banned_areas:
                result_area = vulnerable_area
                min_prob_hold = prob_of_old_area

        if result_area is None:
            print("AI: I not found vulnerable area")
            return None

        print("AI: My vulnerable area is: " + str(result_area.get_name()))
        self.banned_areas.append(result_area)
        return result_area

    def get_neighbours_of_vulnerable_area(self, board: Board, area: Area) -> Optional[Area]:
        neighbours = area.get_adjacent_areas_names()

        for adj in neighbours:
            adjacent_area = board.get_area(adj)

            if adjacent_area.get_owner_name() == self.player_name and \
                    board.is_at_border(adjacent_area) is False and adjacent_area.get_dice() > 1:

                print("AI: My neighbour area, which can support me is: " + str(adjacent_area.get_name()))
                return adjacent_area

            elif adjacent_area.get_owner_name() == self.player_name and \
                    board.is_at_border(adjacent_area) is False:

                if adjacent_area not in self.middle_areas:
                    self.middle_areas.append(adjacent_area)

        print("AI: I not found area, which can support me")
        return None

    def support_middle_areas(self, board: Board, middle_areas: List[Area]) -> Union[tuple[None, None], tuple[Area, Area]]:
        new_middle_areas = []

        for area in middle_areas:
            adjacent_areas = area.get_adjacent_areas_names()

            for adj in adjacent_areas:
                adjacent_area = board.get_area(adj)

                is_owner_name = adjacent_area.get_owner_name() == self.player_name
                is_not_at_border = board.is_at_border(adjacent_area) is False
                is_not_in_middle_areas = adjacent_area not in self.middle_areas
                has_dice = adjacent_area.get_dice() > 1

                if is_owner_name and is_not_at_border is False and has_dice and is_not_in_middle_areas:
                    return area, adjacent_area
                elif is_owner_name and is_not_at_border and is_not_in_middle_areas:
                    if adjacent_area not in new_middle_areas:
                        new_middle_areas.append(adjacent_area)

        if len(new_middle_areas) == 0:
            adjacent_area = None
            area = None
            return area, adjacent_area

        area, adjacent_area = self.support_middle_areas(board, new_middle_areas)
        return area, adjacent_area

