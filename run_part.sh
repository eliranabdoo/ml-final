#!/bin/bash
nohup python -u main.py 8 $1 2>&1 > part$1.out &
