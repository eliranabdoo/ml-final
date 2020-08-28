#!/bin/bash
nohup python -u main.py 16 $1 2>&1 > part$1.out &
