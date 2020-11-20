# Scrambles

A "scramble" is defined as a random sequence of moves that transforms the Rubik's Cube to a non-solved state.

Raw data generated using [TNoodle version 1.0.1](https://www.worldcubeassociation.org/regulations/scrambles/) unless otherwise stated.

Running `get_scrambles.py` generates a `.txt` file with one scramble per line. Scramble length is specified in half turn metric (HTM), but scrambles are converted to quarter turn metric (QTM).

For example: `R' U2 R' F2` is 4 moves in HTM, but its QTM representation is `R' U U R' F F`.

Scrambles found in `full.txt` are 11 moves in HTM, which meets [God's number](https://en.wikipedia.org/wiki/Pocket_Cube#cite_ref-Jaapsch_2-0) for the 2x2x2 cube - the maximum number of turns required to solve the cube.