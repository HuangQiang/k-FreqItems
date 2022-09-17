#!/bin/bash

# ------------------------------------------------------------------------------
#  validate the center representation of frequent items
# ------------------------------------------------------------------------------
g++ -std=c++11 -w -O3 -o freqitem freqitem.cc
m=1000  # number of sampled points as a cluster
t=5     # repeated times

# # ------------------------------------------------------------------------------
# #  URL
# # ------------------------------------------------------------------------------
# n=2396128
# d=3231961
# dname=URL
# input=/nfsdata/DATASET/URL/sparse_bin/1/${dname}_0.bin
# output=results/${dname}

# ./freqitem ${n} ${d} ${m} ${t} ${input} ${output}

# # ------------------------------------------------------------------------------
# #  Criteo10M
# # ------------------------------------------------------------------------------
# n=10000000
# d=1000000
# dname=Criteo10M
# input=/nfsdata/DATASET/Criteo1B/1/${dname}_0.bin 
# output=results/${dname}

# ./freqitem ${n} ${d} ${m} ${t} ${input} ${output}

# ------------------------------------------------------------------------------
#  Avazu
# ------------------------------------------------------------------------------
n=40428960
d=1000000
dname=Avazu
input=/nfsdata/DATASET/Avazu/1/${dname}_0.bin 
output=results/${dname}

./freqitem ${n} ${d} ${m} ${t} ${input} ${output}

# # ------------------------------------------------------------------------------
# #  KDD2012
# # ------------------------------------------------------------------------------
# n=149639104
# d=54686452
# dname=KDD2012
# input=/nfsdata/DATASET/kdd12/1/${dname}_0.bin 
# output=results/${dname}

# ./freqitem ${n} ${d} ${m} ${t} ${input} ${output}
