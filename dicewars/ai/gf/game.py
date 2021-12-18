from typing import Dict, Tuple, Optional

from dicewars.server.area import Area
from dicewars.server.board import Board
from dicewars.server.player import Player

MAX_AREA_COUNT = 30
MAX_PLAYER_COUNT = 4

GameConfiguration = Tuple[int]


def serialise_game_configuration(
        board: Board,
        players: Optional[Dict[int, Player]] = None,
        biggest_regions: Optional[Dict[int, int]] = None,
) -> GameConfiguration:
    """
    Serialises the current game configuration to an integer vector of a static length:
    435 triangle from a matrix of neighbor areas (30x30, but just half)
    30 areas owners
    30 dices counts
    4 biggest regions
    ==
    499 integers
    :param board: The game board.
    :param players: A dictionary of players in the game.
    :param biggest_regions: A dictionary of players and their biggest region's
                            sizes.
    """
    assert players or biggest_regions, 'Given biggest regions directly or by players'
    assert not biggest_regions or len(biggest_regions) == MAX_PLAYER_COUNT, 'Exact count of biggest regions'

    areas: Dict[int, Area] = board.areas
    players = players or dict()

    board_state = []
    # generates triangle matrix of neighbor areas
    for column_area_id in range(1, MAX_AREA_COUNT):
        column_area = areas.get(column_area_id)

        neighbors = column_area.get_adjacent_areas_names() if column_area else {}

        board_state.extend([
            int((col_id + 1) in neighbors)
            for col_id in range(column_area_id, MAX_AREA_COUNT)
        ])

    # generates areas owners
    board_state.extend([
        area.owner_name if (area := areas.get(area_id + 1)) else 0
        for area_id in range(MAX_AREA_COUNT)
    ])

    # generates dices counts
    board_state.extend([
        area.dice if (area := areas.get(area_id + 1)) else 0
        for area_id in range(MAX_AREA_COUNT)
    ])

    # size of biggest region of each player
    if biggest_regions:
        # defined directly (in evaluation time)
        board_state.extend([
            biggest_regions.get(player_id) or 0
            for player_id in range(MAX_PLAYER_COUNT)
        ])
    else:
        # defined by players (in training time)
        board_state.extend([
            p.get_largest_region(board) if (p := players.get(player_id + 1)) else 0
            for player_id in range(MAX_PLAYER_COUNT)
        ])

    return tuple(map(int, board_state))