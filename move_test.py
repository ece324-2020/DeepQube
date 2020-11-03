import cubesim
import numpy as np

c = cubesim.Cube2()
print(c.state)

move_mappping = {
    'F': 0, 'F\'': 1,
    'R': 2, 'R\'': 3,
    'U': 4, 'U\'': 5,
    'L': 6, 'L\'': 7,
    'B': 8, 'B\'': 9,
    'D': 10, 'D\'': 11,
}
while True:
    move = input('Move:')
    c.moves[move_mappping[move]]()
    print(c.state)
