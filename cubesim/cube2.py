import numpy as np
import torch

class Cube2:
    """Cube2 is an implementation of a 2x2x2 cube

    The state representation uses a one-hot encoding per piece, along with an orientation.
    There are eight pieces which may reside in eight locations, with three possible orientations each.

    Each of these one-hot encodings are concatenated to form an 11 wide vector.

    State tensor components:
        `state[0]` - front, top, left
        `state[1]` - front, top, right
        `state[2]` - front, bottom, left
        `state[3]` - front, bottom, right
        `state[4]` - back, top, left
        `state[5]` - back, top, right
        `state[6]` - back, bottom, left
        `state[7]` - back, bottom, right

    The piece encodings are motivated by the pieces in each corner in the  "default orientation"
    of the cube. This is the orientation where green is in front and white is on top.

    The orientations of these pieces are as follows:
        0 - default orientation
        1 - clockwise rotation
        2 - counterclockwise rotation
    """

    def __init__(self):
        self.reset()
        self.moves = [
                self.front, self.front_p,
                self.right, self.right_p,
                self.up, self.up_p,
                self.left, self.left_p,
                self.right, self.right_p,
                self.back, self.back_p,
                self.down, self.down_p,
        ]

    def reset(self):
        pass

    def front(self):
        pass

    def front_p(self):
        pass

    def right(self):
        pass

    def right_p(self):
        pass

    def up(self):
        pass

    def up_p(self):
        pass

    def left(self):
        pass

    def left_p(self):
        pass

    def back(self):
        pass

    def back_p(self):
        pass

    def down(self):
        pass

    def down_p(self):
        pass
