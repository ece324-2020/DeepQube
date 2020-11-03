import numpy as np
import torch


class Cube2:
    """Cube2 is an implementation of a 2x2x2 cube

    Interally we only keep track of sticker positions.
    """

    """ The `face_mapping` dictionary maps a LFRBTD to it's index in the internal state array."""
    face_mapping = {'L': 0, 'F': 1, 'R': 2, 'B': 3, 'U': 4, 'D': 5}

    """ The `adj_lookup` dictionary maps a LFRBTD to its adjacent faces for rotation.
    The tuples are as follows `(face, row_index, column_index, flip)`.
    The `row_index` and `column_index` indicate which parts of the face need to be copied onto the next face at its specific indicies.
    The `flip` boolean indicates that the order of the moved parts need flipped when copying from/to that face.
    See `__rotate` for how this is used.
    """
    adj_lookup = {
            'L': [('U', slice(None), 0, True), ('F', slice(None), 0, False), ('D', slice(None), 0, False), ('B', slice(None), 1, True)],
            'F': [('U', 1, slice(None), True), ('R', slice(None), 0, False), ('D', 0, slice(None), True), ('L', slice(None), 1, False)],
            'R': [('U', slice(None), 1, False), ('B', slice(None), 0, True), ('D', slice(None), 1, True), ('F', slice(None), 1, False)],
            'B': [('U', 0, slice(None), False), ('L', slice(None), 0, True), ('D', 1, slice(None), False), ('R', slice(None), 1, True)],
            'U': [('F', 0, slice(None), False), ('L', 0, slice(None), False), ('B', 0, slice(None), False), ('R', 0, slice(None), False)],
            'D': [('F', 1, slice(None), False), ('R', 1, slice(None), False), ('B', 1, slice(None), False), ('L', 1, slice(None), False)],
    }

    def __init__(self):
        self.reset()
        self.moves = [
                self.front, self.front_p,
                self.right, self.right_p,
                self.up, self.up_p,
                self.left, self.left_p,
                self.back, self.back_p,
                self.down, self.down_p,
        ]

    """This function converts the internal state to an embedding to be passed into the Neural Network.
    The embedding uses a one-hot encoding per piece, along with an orientation.
    There are eight pieces which may reside in eight locations, with three possible orientations each.

    Each of these one-hot encodings are concatenated to form an 11 wide vector.

    State tensor components:
        `embedding[0]` - front, top, left
        `embedding[1]` - front, top, right
        `embedding[2]` - front, down, left
        `embedding[3]` - front, down, right
        `embedding[4]` - back, top, left
        `embedding[5]` - back, top, right
        `embedding[6]` - back, down, left
        `embedding[7]` - back, down, right

    The piece encodings are motivated by the pieces in each corner in the "default orientation"
    of the cube. This is the orientation where green is in front and white is on top.

    The orientations of these pieces are as follows:
        0 - default orientation
        1 - clockwise rotation
        2 - counterclockwise rotation
    """
    def get_embedding(self):
        pass

    def reset(self):
        self.state = np.array([np.tile(i, (2, 2)) for i in range(6)])

    def front(self):
        self.__rotate('F', prime=False)

    def front_p(self):
        self.__rotate('F', prime=True)

    def right(self):
        self.__rotate('R', prime=False)

    def right_p(self):
        self.__rotate('R', prime=True)

    def up(self):
        self.__rotate('U', prime=False)

    def up_p(self):
        self.__rotate('U', prime=True)

    def left(self):
        self.__rotate('L', prime=False)

    def left_p(self):
        self.__rotate('L', prime=True)

    def back(self):
        self.__rotate('B', prime=False)

    def back_p(self):
        self.__rotate('B', prime=True)

    def down(self):
        self.__rotate('D', prime=False)

    def down_p(self):
        self.__rotate('D', prime=True)

    def __rotate(self, face, prime=False):
        # There are two steps to each rotation
        # 1. Rotate the face.
        #   1 2       3 1
        #   3 4       4 2
        # 2. Identify the adjacent elements to rotate, and rotate them.
        #   F => T[1,:] -> R[:,0] -> D[0,:] -> L[1,:] -> T[1,:]
        #   This is stored in adj_lookup

        self.state[self.face_mapping[face]] = np.rot90(self.state[self.face_mapping[face]], -1 if not prime else 1)

        adj_elements = self.adj_lookup[face]
        it = range(0, len(adj_elements))
        if prime:
            it = reversed(it)

        a = adj_elements[-1 if not prime else 0]
        old = self.state[self.face_mapping[a[0]]][a[1:3]]
        for i in it:
            b = adj_elements[i]
            flip = a[3] if prime else b[3]

            temp = np.copy(self.state[self.face_mapping[b[0]]][b[1:3]])
            self.state[self.face_mapping[b[0]]][b[1:3]] = old if not flip else np.flip(old)
            old = temp
            a = b

