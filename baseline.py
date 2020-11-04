import numpy as np
import time
import argparse
import cubesim
from cubesim.visualizer import print_cube

# implements a hard-coded naive baseline model

parser = argparse.ArgumentParser()
parser.add_argument("animate", help="\'y\' or \'n\'")
parser.add_argument("scramble")
args = parser.parse_args()

c = cubesim.Cube2()
sleep_time = 0 if args.animate == 'n' else 1

oll_algorithms = {
    'h': 'R R U U R U U R R',
    'pi': 'R U U R R U\' R R U\' R R U U R',
    'antisune': 'R\' U\' R U\' R\' U U R',
    'sune': 'R U R\' U R U U R\'',
    'l': 'F R\' F\' R U R U\' R\'',
    't': 'R U R\' U\' R\' F R F\'',
    'u': 'F R U R\' U\' F\''
}

pll_algorithms = {
    'y': 'R U\' R\' U\' F F U\' R U R\' D R R',
    'j': 'R U U R\' U\' R U U L\' U R\' U\' L',
}

insert = {
    'fr': 'R U R\' U\'',
    'fl': 'L\' U\' L U',
    'br': 'R\' U\' R U',
    'bl': 'L U L\' U\''
}

sledge = {
    'fr': 'R\' F R F\'',
    'fl': 'L F\' L\' F',
    'br': 'R B\' R\' B',
    'bl': 'L\' B L B\''
}


def solve_layer1(cube):

    check_map_top = {
        3: 1,
        1: 2,
        0: 3,
        2: 0
    }

    check_map_side_0 = {
        0: 3,
        1: 0,
        2: 1,
        3: 2
    }

    check_map_side_1 = {
        3: 0,
        0: 1,
        1: 2,
        2: 3
    }

    can_move = False

    for f, face in enumerate(cube.state):
        flat = np.ravel(face)
        for s, square in enumerate(flat):
            if square == 5:
                # yellow sticker is on sides (can solve)
                if f in (0, 1, 2, 3) and s in (0, 1):
                    can_move = True
                    if s == 0:
                        # YRG
                        if cube.state[check_map_side_0[f]][0][1] == 1:
                            if f == 1:
                                turn_p(cube)
                            elif f == 3:
                                turn(cube)
                            elif f == 0:
                                turn(cube)
                                turn(cube)

                            do_algorithm(cube, insert['fr'])

                        # YOG
                        elif cube.state[check_map_side_0[f]][0][1] == 0:
                            if f == 0:
                                turn_p(cube)
                            elif f == 2:
                                turn(cube)
                            elif f == 3:
                                turn(cube)
                                turn(cube)

                            do_algorithm(cube, sledge['fl'])

                        # YOB
                        elif cube.state[check_map_side_0[f]][0][1] == 3:
                            if f == 3:
                                turn_p(cube)
                            elif f == 1:
                                turn(cube)
                            elif f == 2:
                                turn(cube)
                                turn(cube)

                            do_algorithm(cube, insert['bl'])

                        # YRB
                        elif cube.state[check_map_side_0[f]][0][1] == 2:
                            if f == 2:
                                turn_p(cube)
                            elif f == 0:
                                turn(cube)
                            elif f == 1:
                                turn(cube)
                                turn(cube)

                            do_algorithm(cube, sledge['br'])
                    elif s == 1:
                        # YRG

                        if cube.state[check_map_side_1[f]][0][0] == 2:
                            if f == 0:
                                turn_p(cube)
                            elif f == 2:
                                turn(cube)
                            elif f == 3:
                                turn(cube)
                                turn(cube)

                            do_algorithm(cube, sledge['fr'])
                        # YOG
                        elif cube.state[check_map_side_1[f]][0][0] == 1:
                            if f == 3:
                                turn_p(cube)
                            elif f == 1:
                                turn(cube)
                            elif f == 2:
                                turn(cube)
                                turn(cube)

                            do_algorithm(cube, insert['fl'])

                        # YOB
                        elif cube.state[check_map_side_1[f]][0][0] == 0:
                            if f == 2:
                                turn_p(cube)
                            elif f == 0:
                                turn(cube)
                            elif f == 1:
                                turn(cube)
                                turn(cube)

                            do_algorithm(cube, sledge['bl'])
                        # YRB
                        elif cube.state[check_map_side_1[f]][0][0] == 3:
                            if f == 1:
                                turn_p(cube)
                            elif f == 3:
                                turn(cube)
                            elif f == 0:
                                turn(cube)
                                turn(cube)

                            do_algorithm(cube, insert['br'])

                # yellow stick is on the top (can solve)
                elif f == 4:
                    can_move = True
                    # YRG
                    if cube.state[check_map_top[s]][0][1] == 2:
                        if s == 1:
                            turn(cube)
                        elif s == 0:
                            turn(cube)
                            turn(cube)
                        elif s == 2:
                            turn_p(cube)

                        for i in range(3):
                            do_algorithm(cube, insert['fr'])
                    # YOG
                    elif cube.state[check_map_top[s]][0][1] == 1:
                        if s == 3:
                            turn(cube)
                        elif s == 0:
                            turn_p(cube)
                        elif s == 1:
                            turn(cube)
                            turn(cube)

                        for i in range(3):
                            do_algorithm(cube, insert['fl'])
                    # YRB
                    elif cube.state[check_map_top[s]][0][1] == 3:
                        if s == 0:
                            turn(cube)
                        elif s == 1:
                            turn_p(cube)
                        elif s == 2:
                            turn(cube)
                            turn(cube)

                        for i in range(3):
                            do_algorithm(cube, insert['br'])
                    # YOB
                    elif cube.state[check_map_top[s]][0][1] == 0:
                        if s == 2:
                            turn(cube)
                        elif s == 1:
                            turn_p(cube)
                        elif s == 3:
                            turn(cube)
                            turn(cube)

                        for i in range(3):
                            do_algorithm(cube, insert['bl'])

    if can_move is False:
        if cube.state[0][1][1] == 5 or cube.state[1][1][0] == 5:
            do_algorithm(cube, insert['fl'])
        elif cube.state[1][1][1] == 5 or cube.state[2][1][0] == 5:
            do_algorithm(cube, insert['fr'])
        elif cube.state[2][1][1] == 5 or cube.state[3][1][0] == 5:
            do_algorithm(cube, insert['br'])
        elif cube.state[3][1][1] == 5 or cube.state[0][1][0] == 5:
            do_algorithm(cube, insert['bl'])

        elif cube.state[5][0][0] == 5 and not piece_solved(cube, 'fl'):
            do_algorithm(cube, insert['fl'])

        elif cube.state[5][0][1] == 5 and not piece_solved(cube, 'fr'):
            do_algorithm(cube, insert['fr'])

        elif cube.state[5][1][0] == 5 and not piece_solved(cube, 'bl'):
            do_algorithm(cube, insert['bl'])

        elif cube.state[5][1][1] == 5 and not piece_solved(cube, 'br'):
            do_algorithm(cube, insert['br'])


def piece_solved(cube, piece):
    if piece == 'br':
        if cube.state[2][1][1] == 2 and cube.state[3][1][0] == 3:
            return True
    elif piece == 'bl':
        if cube.state[0][1][0] == 0 and cube.state[3][1][1] == 3:
            return True
    elif piece == 'fr':
        if cube.state[1][1][1] == 1 and cube.state[2][1][0] == 2:
            return True
    elif piece == 'fl':
        if cube.state[0][1][1] == 0 and cube.state[1][1][0] == 1:
            return True

    return False


def do_algorithm(cube, alg):
    for move in alg.split(' '):
        cube.moves[cube.move_mappping[move]]()
        time.sleep(sleep_time)
        if sleep_time == 1:
            print('\33[37m' + move + '\33[37m')
            print_cube(cube.state)


def turn_p(cube):
    cube.moves[cube.move_mappping['U\'']]()
    time.sleep(sleep_time)
    if sleep_time == 1:
        print('\33[37m' + 'U\'' + '\33[37m')
        print_cube(cube.state)


def turn(cube):
    cube.moves[cube.move_mappping['U']]()
    time.sleep(sleep_time)
    if sleep_time == 1:
        print('\33[37m' + 'U' + '\33[37m')
        print_cube(cube.state)


def oll_layer2(cube):
    face = np.ravel(cube.state[4])
    num_oriented = np.sum(face == 4)

    if num_oriented == 2:
        if (cube.state[4][0][1] == 4 and cube.state[4][1][1] == 4):
            if cube.state[0][0][0] == 4 and cube.state[0][0][1] == 4:
                do_algorithm(cube, oll_algorithms['u'])
            elif cube.state[1][0][0] == 4:
                do_algorithm(cube, oll_algorithms['t'])

        elif (cube.state[4][0][0] == 4
              and cube.state[4][1][1] == 4
              and cube.state[1][0][0] == 4):
            do_algorithm(cube, oll_algorithms['l'])
        else:
            turn(cube)
    elif num_oriented == 1:
        if (cube.state[1][0][1] == 4
                and cube.state[2][0][1] == 4
                and cube.state[3][0][1] == 4):
            do_algorithm(cube, oll_algorithms['sune'])

        elif (cube.state[1][0][0] == 4
                and cube.state[2][0][0] == 4
                and cube.state[3][0][0] == 4):
            do_algorithm(cube, oll_algorithms['antisune'])

        else:
            turn(cube)
    elif num_oriented == 0:
        if (cube.state[1][0][0] == 4
            and cube.state[1][0][1] == 4
                and cube.state[3][0][0] == 4
                and cube.state[3][0][1] == 4):
            do_algorithm(cube, oll_algorithms['h'])
        elif (cube.state[0][0][0] == 4
              and cube.state[0][0][1] == 4
              and cube.state[1][0][1] == 4):
            do_algorithm(cube, oll_algorithms['pi'])
        else:
            turn(cube)


def pll_layer2(cube):

    if (np.absolute(cube.state[2][0][1] - cube.state[2][0][0]) == 2
            and cube.state[0][0][1] == cube.state[0][0][0]):
        do_algorithm(cube, pll_algorithms['j'])

    elif (np.absolute(cube.state[2][0][1] - cube.state[2][0][0]) == 2
            and np.absolute(cube.state[0][0][1] - cube.state[0][0][0]) == 2):
        do_algorithm(cube, pll_algorithms['y'])

    else:
        turn(cube)


def is_solved(cube):
    solved = True
    for face in cube.state:
        if not np.all(face == face[0][0]):
            solved = False

    return solved


def bottom_is_solved(cube):
    solved = False
    if (np.all(cube.state[5] == cube.state[5][0][0])
            and cube.state[0][1][0] == cube.state[0][1][1]
            and cube.state[1][1][0] == cube.state[1][1][1]
            and cube.state[2][1][0] == cube.state[2][1][1]):
        solved = True

    return solved


def top_is_solved(cube):
    solved = False
    if np.all(cube.state[4] == 4):
        solved = True

    return solved


def post_process_string(s):
    s = s.replace('U U U', 'U\'')
    s = s.replace('U U', 'U2')
    s = s.replace('R R', 'R2')
    s = s.replace('F F', 'F2')
    s = s.replace('L L', 'L2')
    s = s.replace('B B', 'B2')
    s = s.replace('D D', 'D2')
    return s


def baseline_solver():
    c.load_scramble(args.scramble)
    print_cube(c.state)

    while bottom_is_solved(c) is False:
        solve_layer1(c)

    while top_is_solved(c) is False:
        oll_layer2(c)

    while is_solved(c) is False:
        pll_layer2(c)

    scramble = args.scramble.split(' ')

    soln1 = c.history[len(scramble):]
    soln2 = post_process_string(' '.join(soln1))
    soln3 = soln2.split()

    print('\33[37m' + 'scramble:',
          post_process_string(args.scramble) + '\33[37m')
    print('\33[37m' + f'solution ({len(soln3)}): ', ''.join(soln2) + '\33[37m')

    print_cube(c.state)


baseline_solver()
