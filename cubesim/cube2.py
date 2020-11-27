import numpy as np
import torch


SOLVED = np.array([np.tile(i, (2, 2)) for i in range(6)])


class Cube2:
    """Cube2 is an implementation of a 2x2x2 cube

    Interally we only keep track of sticker positions.
    """

    """ The `face_mapping` dictionary maps a LFRBTD to it's index in the internal state array.
    The nature of the internal state allows us to use this index as the colour associated with that face on the cube.
    """
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

    """ This dictionary maps a move to an index into the `moves` list"""
    move_mappping = {
        'F': 0, 'F\'': 1,
        'R': 2, 'R\'': 3,
        'U': 4, 'U\'': 5,
        'L': 6, 'L\'': 7,
        'B': 8, 'B\'': 9,
        'D': 10, 'D\'': 11,
    }

    def __init__(self):
        self.reset()
        self.history = []
        self.embedding_dim = (24, 6)
        # the order of this list matters
        self.moves = (
            self.front, self.front_p,
            self.right, self.right_p,
            self.up, self.up_p,
            self.left, self.left_p,
            self.back, self.back_p,
            self.down, self.down_p,
        )

    def get_embedding(self, device='cpu'):
        """The embedding is based on the sticker representation.
        The cube embedding is a 24x6 tensor, where the 24 consists of a flattening
        of a 6x2x2 tensor, and the 6 is a one-hot encoding of the colours.
        """
        embedding = torch.zeros(self.embedding_dim, device=device)
        for i, colour in enumerate(self.state.flat):
            embedding[i][colour] = 1
        return embedding.view(-1)

    def load_scramble(self, s):
        """Loads the scramble contained in the string `s`"""
        for move in s.split(' '):
            self.moves[self.move_mappping[move]]()

    def load_state(self, s):
        self.state = np.array(list(map(int, s.split(' ')))).reshape(6, 2, 2)

    def is_solved(self):
        return np.array_equal(self.state, SOLVED)

    def reset(self):
        self.state = np.copy(SOLVED)

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

        self.history.append(face + ('' if prime is False else '\''))

        self.state[self.face_mapping[face]] = np.rot90(
            self.state[self.face_mapping[face]], -1 if not prime else 1)

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
