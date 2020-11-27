import functools
import itertools
import sys

sys.path.insert(0, '../') # Hack to get cubesim import working

import cubesim

def generate_solution(scramble):
    cube = cubesim.Cube2()
    cube.load_scramble(scramble)

    lines = []
    for move in reversed(scramble.split(' ')):
        # reverse the move itself
        if move[-1] == "'":
            move = cube.move_mappping[move[:-1]]
        else:
            move = cube.move_mappping[move + "'"]

        lines.append(f'{serialize_state(cube)}\t{move}')
    return lines


def serialize_state(cube):
    return ' '.join(map(str, cube.state.reshape(-1).tolist()))

if __name__ == '__main__':

    with open('./1layer.txt', 'r') as f:
        lines = f.readlines()

    output = itertools.chain(*map(generate_solution, map(lambda line: line.strip(), lines)))

    with open('./supervised_data.tsv', 'w') as f:
        f.writelines(map(lambda l: f'{l}\n', output))


