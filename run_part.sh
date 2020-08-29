#!/bin/bash
nohup python -u main.py 8 $1 2>&1 > baddbs1_part$1.out &
