#!/usr/bin/env python3

import argparse, sys, math

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('dst', type=str)
ARGS = vars(parser.parse_args())

f_src = open(ARGS['src'])
f_dst = open(ARGS['dst'], 'w')

first_label = int(next(f_src).split()[1])

for line in f_src:
    if first_label == 0:
        prd = 1 - float(line.split()[1])
    else:
        prd = float(line.split()[1])
    f_dst.write('{0}\n'.format(prd))
