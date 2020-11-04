#!/bin/bash

while read line; do python3 baseline.py n "$line"; done < data/10moves.txt
