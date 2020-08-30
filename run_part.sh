#!/bin/bash
nohup python -u main_solve_bad_dbs.py 4 $1 2>&1 > baddbs1_part$1.out &
