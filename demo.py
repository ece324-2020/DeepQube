import argparse
import torch
import webbrowser
from inference import solve

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate the solution to a scramble')
    parser.add_argument('scramble', type=str, help='path to file with scrambles to test')
    parser.add_argument('--model1', type=str, help='path to the reinforcement learning model', default='model1.pt')
    parser.add_argument('--model2', type=str, help='path to the supervised learning model', default='model2.pt')

    args = parser.parse_args()
    model1 = torch.load(args.model1, map_location='cpu')
    model2 = torch.load(args.model2, map_location='cpu')

    scramble_qtm = args.scramble.replace('U2', 'U U').replace('R2', 'R R').replace('F2', 'F F')

    is_solved, solution = solve(model1, model2, scramble_qtm, 40)

    solution_str = ''

    move_mapping = {
        0: 'F', 1: 'F-',
        2: 'R', 3: 'R-',
        4: 'U', 5: 'U-',
        6: 'L', 7: 'L-',
        8: 'B', 9: 'B-',
        10: 'D', 11: 'D-',
    }

    for move in solution:
        solution_str = solution_str + move_mapping[move.item()] + "_"

    scramble_str = args.scramble.replace("'", "-").replace(" ", "_")

    if is_solved:
        webbrowser.open(f'https://alg.cubing.net/?setup={scramble_str}&alg={solution_str[:-1]}&puzzle=2x2x2')
    else:
        print('Error: DeepQube cannot solve this scramble')
