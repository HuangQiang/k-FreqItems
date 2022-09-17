#!/bin/bash

# # ------------------------------------------------------------------------------
# #  convert sparse data sets (with svm format) into sparse format
# # ------------------------------------------------------------------------------
# g++ -std=c++11 -w -O3 -o sparse sparse.cc

# # KDD2012
# n=149639105
# d=54686452
# input="sparse_data/KDD2012/kdd12"
# output="../sparse_bin/KDD2012_0.bin"
# ./sparse ${n} ${d} ${input} ${output}

# # ------------------------------------------------------------------------------
# #  split sparse data sets (bin format) into m equal partitions
# # ------------------------------------------------------------------------------
# g++ -std=c++11 -w -O3 -o partition partition.cc

# # URL
# N=2396130
# n=2396128
# dname=URL
# ./partition ${N} ${n} ${dname}

# # KDD2012
# N=149639105
# n=149639104
# dname=KDD2012
# ./partition ${N} ${n} ${dname}

# # Criteo10M
# N=178274637
# n=10000000
# dname=Criteo10M
# ./partition ${N} ${n} ${dname}

# # Criteo100M
# N=178274637
# n=100000000
# dname=Criteo100M
# ./partition ${N} ${n} ${dname}

# ------------------------------------------------------------------------------
#  Convert Avazu into Sparse Format with 1,2,4,8 partitions
# ------------------------------------------------------------------------------
# # convert sparse data sets (with svm format) into sparse format
# g++ -std=c++11 -w -O3 -o avazu avazu.cc

# n1=14596137
# n2=25832830
# d=1000000
# input1="sparse_data/Avazu/avazu-app"
# input2="sparse_data/Avazu/avazu-site"
# output="../sparse_bin/Avazu_0.bin"
# ./avazu ${n1} ${n2} ${d} ${input1} ${input2} ${output}

# split sparse data sets (bin format) into m equal partitions
g++ -std=c++11 -w -O3 -o partition partition.cc

N=40428967
n=40428960
dname=Avazu
./partition ${N} ${n} ${dname}