pre_moves = ['U ', 'U\' ', 'U U ', '']
post_moves = [' U', ' U\'', ' U U', '']

reverse_mapping = {
    'U': 'U\'',
    'D': 'D\'',
    'R': 'R\'',
    'L': 'L\'',
    'B': 'B\'',
    'F': 'F\'',
    'U\'': 'U',
    'D\'': 'D',
    'R\'': 'R',
    'L\'': 'L',
    'B\'': 'B',
    'F\'': 'F',
}

cases = [
    "R\' U\' R U\' R\' U2 R",
    "R U2 R\' F R\' F\' R U\' R U\' R\'",
    "F\' L F L\' U2 L\' U2 L",
    "R\' F R F\' R U R\'",
    "R U2 R\' U2 R\' F R F\'",
    "R2 F R U2 R U\' R\' U2 F\' R",
    "R2 U2 R U2 R2",
    "R\' F R F\' R U R2 F R F\' R U R\'",
    "R U R\' U R U R\' F R\' F\' R",
    "F R2 U\' R2 U\' R2 U R2 F\'",
    "F\' R U R\' U\' R\' F R",
    "F R\' F\' R U R U\' R\'",
    "R U2 R2 F R F\' R U2 R\'",
    "R\' U R\' U2 R U\' R\' U R U\' R2",
    "R U\' R\' U R U\' R\' F R\' F\' R2 U R\'",
    "R\' F\' R U R\' U\' R\' F R2 U\' R\' U2 R",
    "F R U R\' U\' R U R\' U\' F\'",
    "R2 U R\' U\' F R F\' R U\' R2",
    "R U\' R\' F R\' F R U R\' F R",
    "R U\' R U\' R\' U R\' F R2 F\'",
    "R U2 R\' U\' R U R\' U2 R\' F R F\'",
    "R\' F2 R F\' U2 R U\' R\' U\' F",
    "R U R\' U R U2 R\'",
    "R\' F R2 F\' R U2 R\' U\' R2",
    "F R\' F\' R U2 R U2 R\'",
    "R U\' R\' F R\' F\' R",
    "R U\' R U\' R\' U R\' U\' F R\' F\'",
    "L\' U2 L U2 L F\' L\' F",
    "R U R\' U\' R\' F R F\'",
    "L\' U\' L U L F\' L\' F",
    "F U\' R U2 R\' U\' F2 R U R\'",
    "R\' U R\' F U\' R U F2 R2",
    "F R U R\' U\' R U\' R\' U\' R U R\' F\'",
    "R\' U R U2 R2 F R F\' R",
    "F R U R\' U\' F\'",
    "R2 F2 R U R\' F U\' R U R2",
    "F R U R\' U2 F\' R U\' R\' F",
    "R2 F R F\' R\' F2 R U R\' F R2",
    "R U\' R2 F R F\' R U R\' U\' R U R\'",
    "R\' U R\' F R F\' R U2 R\' U R",
    "R U R\' F\' R U R\' U\' R\' F R2 U\' R\'",
    "F R U\' R\' U\' R U R\' F\' R U R\' U\' R\' F R F\'"
]


def inverse_alg(alg):
    temp = alg.replace('U2', 'U U')
    temp = temp.replace('D2', 'D D')
    temp = temp.replace('R2', 'R R')
    temp = temp.replace('L2', 'L L')
    temp = temp.replace('F2', 'F F')
    temp = temp.replace('B2', 'B B')
    moves = temp.split()
    inverse = []
    for i in reversed(moves):
        inverse.append(reverse_mapping[i])

    return ' '.join(inverse)


f = open("1layer.txt", "w")

for case in cases:
    alg = inverse_alg(case)
    for i in pre_moves:
        for j in post_moves:
            f.write(i + alg + j + '\n')
