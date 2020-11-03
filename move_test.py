import cubesim
import numpy as np

c = cubesim.Cube2()

color_map = {
    2: '\33[31m',
    1: '\33[92m',
    0: '\33[33m',
    3: '\33[34m',
    5: '\33[93m',
    4: '\33[37m'
}


def print_square(value, j):
    if j == 0:
        print(color_map[value] + str(value) + ' ' + color_map[value], end="")
    else:
        print(color_map[value] + str(value) + ' ' + color_map[value])


def print_face(state, face: int):
    for i in range(2):
        print("    ", end="")
        for j in range(2):
            print_square(state[face][i][j], j)


def print_faces(state):
    for i in range(2):
        for j in range(2):
            print_square(state[0][i][j], 0)
        for j in range(2):
            print_square(state[1][i][j], 0)
        for j in range(2):
            print_square(state[2][i][j], 0)
        for j in range(2):
            print_square(state[3][i][j], j)


def print_cube(state):

    print_face(state, 4)

    print_faces(state)

    print_face(state, 5)


print_cube(c.state)

move_mappping = {
    'F': 0, 'F\'': 1,
    'R': 2, 'R\'': 3,
    'U': 4, 'U\'': 5,
    'L': 6, 'L\'': 7,
    'B': 8, 'B\'': 9,
    'D': 10, 'D\'': 11,
}

while True:
    move = input('\33[37m' + 'Move:' + '\33[37m')
    c.moves[move_mappping[move]]()
    print_cube(c.state)
