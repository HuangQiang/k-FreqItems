#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  basic parameters
# ------------------------------------------------------------------------------
dname=URL                           # data set name
d=3231961                           # dimension
F=int32                             # data format: uint8, uint16, int32, float32
GA=0.5                              # global alpha
LA=0.3                              # local  alpha
max_iter=10                         # maximum iteration

GB=0.9                              # global beta
LB=0.8                              # local  beta
b=1                                 # threshold of #bucket IDs in a bin
s=3                                 # seeding algorithm (0-3, 3 is silk)

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 4 GPUs
# ------------------------------------------------------------------------------
n=599032                            # number of data points
P=/nfsdata/DATASET/URL/sparse_bin/4/${dname} # prefix for data set
O=results/${dname}/4/               # output folder

for m1 in 16 20 24 28 
do 
  for h1 in 4 6 
  do 
    for t in 30 40 50
    do 
      for m2 in 8 10 12
      do 
        for h2 in 4 6 
        do 
          for delta in 30 50 
          do 
            mpirun -n 4 -hostfile hosts ./silk -alg 0 -n ${n} -d ${d} \
              -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} -b ${b} -D ${delta} \
              -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
          done
        done
      done
    done
  done
done
