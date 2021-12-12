import random
import logging
from typing import List, Union, Tuple, Deque, Dict
import copy

from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.client.game.board import Board


class AI:

    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.cache = []
        self.sum_of_dices = 0
        self.promising_attack = None

    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        print("----------------------------------------------------------------------------------------")
        print("------- INITIAL BOARD -----")
        print(len(board.get_player_areas(self.player_name)))
        print("Start geffik AI Player-" + str(self.player_name) + " turn")
        self.promising_attack = []
        self.sum_of_dices = 0
        self.actions_calculation(board, 2)

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
            source = attack[0]
            target = attack[1]

            print("INITIAL BOARD IN GAME " + str(len(board.get_player_areas(self.player_name))))
            tmp = 0
            if source.get_dice() > 1:
                tmp = probability_of_successful_attack(board, source.get_name(), target.get_name())
            print("My probability on successfull attack is: " + str(tmp))

            if tmp < 0.30:
                print("AI: Probability is too small, i will search for better options")
                continue

            board_after_simulation = copy.deepcopy(board)
            board_after_simulation = self.simulate_turn(board_after_simulation, source, target)

            print("INITIAL BOARD AFTER SIMULATION " + str(len(board.get_player_areas(self.player_name))))
            print("AI: Probability is ok, i will simulate attack and then search deeper in depth: " + str(depth - 1))
            number_of_dices = self.actions_calculation(board_after_simulation, depth - 1)
            print("AI: Back in depth: " + str(depth))

            if len(self.cache) < depth:
                self.cache.append(number_of_dices)
                print("Adding biggest num of dices to cache (1. condition): " + str(number_of_dices))
            else:
                if number_of_dices > self.cache[depth - 1]:
                    self.cache[depth - 1] = number_of_dices
                    print("Adding biggest num of dices to cache (2. condition): " + str(number_of_dices))

            if len(self.cache) > 0:
                tmp2 = 0
                for item in self.cache:
                    tmp2 += item
                print("Cache was summed: " + str(tmp2))

                if tmp2 > self.sum_of_dices:
                    self.sum_of_dices = tmp2
                    self.promising_attack = attack
                    print("I found better option, so i replaced current, storing attack source: " + str(self.promising_attack[0].get_name()) +
                          " target: " + str(self.promising_attack[1].get_name()))

            if depth == 2:
                self.cache = []

        print("I already research all my possible attacks in depth: " + str(depth))
        print("Return from recursion with number of dices: " + str(number_of_dices))
        return number_of_dices

    def get_player_number_of_dice(self, board):
        num_of_dice = board.get_player_dice(self.player_name)
        return num_of_dice

    def simulate_turn(self, board, source, target):
        """Simulation of turn on given game board
        Attributes
        ----------
        board : board
        source : int
        target : int
        Returns
        -------
        board
        """
        src_name = source.get_name()
        src_tmp = board.get_area(src_name)

        tgt_name = target.get_name()
        tgt_tmp = board.get_area(tgt_name)

        tgt_tmp.set_owner(source.get_owner_name())
        tgt_tmp.set_dice(source.get_dice() - 1)

        src_tmp.set_dice(1)

        return board

