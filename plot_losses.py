#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    filename = 'losses' if len(sys.argv) == 0 else sys.argv[1]
    episode, loss, acc = np.loadtxt(filename, skiprows=1, unpack=True)

    plt.subplot(2, 1, 1)
    plt.plot(episode, loss)
    plt.subplot(2, 1, 2)
    plt.plot(episode, acc)
    plt.show()

