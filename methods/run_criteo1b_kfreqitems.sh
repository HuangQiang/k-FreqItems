#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  basic parameters
# ------------------------------------------------------------------------------
dname=Criteo1B                      # data set name
d=1000000                           # dimension
F=int32                             # data format: uint8, uint16, int32, float32
GA=0.2                              # global alpha
LA=0.1                              # local  alpha
max_iter=10                         # maximum iteration

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 4 GPUs
# ------------------------------------------------------------------------------
n=250000000                         # number of data points
P=/nfsdata/DATASET/Criteo1B/4/${dname} # prefix for data set
O=results/${dname}/4/               # output folder

for s in 0 1 2
do 
  for k in 10 1000 2000 4000 6000 8000 10000
  do
    mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
      -m ${max_iter} -s ${s} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
  done
done
