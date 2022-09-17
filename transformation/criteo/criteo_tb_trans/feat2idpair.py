#!/usr/bin/env python3

import argparse, csv, sys, math, hashlib

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('--I', action="store_true")
parser.add_argument('feat_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

def feat_enc(j, value):
    res = 0
    if j <= 13:
        # I case
        if value != '':
            value = int(value)
            res = j + ((value + 1) << 7)
        else:
            res = j
    elif 14 <= j:
        # C case
        if value != '':
            if value == "less":
                res = j + (1<<60)
            else:
                value = int(value, 16)
                res = j + ((value + 1) << 7)
        else:
            res = j
    return res

def feat_hsh(j, value):
    key = 0
    if j <= 13:
        # I case
        field = 'I' + str(j)
        if value != '':
            value = int(value)
            if value > 2:
                value = int(math.log(float(value))**2)
            else:
                value = 'SP'+str(value)
        key = field + '-' + str(value)
    elif 14 <= j:
        # C case
        field = 'C' + str(j-13)
        if value == "less":
            key = field+"less"
        else:
            key = field + '-' + value
    return (key, hashstr(key, args['nr_bins']))

fi = open(args['feat_path'], 'r')
fo = open(args['out_path'], 'w')

print(args['feat_path'], args['out_path'], args['I'])

for line in fi.readlines():
    col, feat, cnt = 0, '', '0'
    words = line.split()
    if len(words) == 2:
        col, cnt = words
    elif len(words) == 3:
        col, feat, cnt = words
    else:
        print("WTF?")
        print(line)
        continue
    col = int(col)
    cnt = int(cnt)
    if cnt < args['threshold']:
        continue
    (key, hsh) = feat_hsh(col, feat)
    if not args['I'] or col <= 13:
        fo.write("%d %d\n" % (feat_enc(col, feat), hsh))
