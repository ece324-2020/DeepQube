import sys

sys.path.insert(0, '../')  # Hack to get cubesim import working

import cubesim

def serialize_state(cube):
    return ' '.join(map(str, cube.state.reshape(-1).tolist()))


if __name__ == '__main__':
    with open('./1layer.txt', 'r') as f:
        lines = f.readlines()

    states = {}
    for line in lines:
        scramble = line.strip()

        cube = cubesim.Cube2()
        cube.load_scramble(scramble)

        for move in reversed(scramble.split(' ')):
            state = serialize_state(cube)
            if state in states:
                break  # we've seen how to solve this state before

            # reverse the move itself
            if move[-1] == "'":
                move = cube.move_mapping[move[:-1]]
            else:
                move = cube.move_mapping[move + "'"]

            cube.moves[move]()
            states[state] = move

    output = map(lambda item: f'{item[0]}\t{item[1]}\n', states.items())

    with open('./supervised_data.tsv', 'w') as f:
        f.writelines(output)
