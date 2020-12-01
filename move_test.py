#!/usr/bin/env python

import cubesim
from cubesim.visualizer import print_cube

c = cubesim.Cube2()
print_cube(c.state)

while True:
    move = input('\33[37m' + 'Move:' + '\33[37m')
    c.moves[c.move_mapping[move]]()
    print_cube(c.state)
