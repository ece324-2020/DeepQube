#!/bin/bash

while read line; do python3 baseline.py --animate y "$line"; done < data/10moves.txt
