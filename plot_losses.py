#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = []
    filename = 'losses' if len(sys.argv) == 0 else sys.argv[1]

    f = open(filename, 'r')
    for l in f.readlines():
        try:
            data.append(float(l))
        except:
            pass

    plt.plot(data)
    plt.show()

