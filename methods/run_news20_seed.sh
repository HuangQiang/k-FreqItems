#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  basic parameters
# ------------------------------------------------------------------------------
dname=News20                        # data set name
d=62061                             # dimension
F=int32                             # data format: uint8, uint16, int32, float32
GA=0.2                              # global alpha
LA=0.2                              # local  alpha
max_iter=10                         # maximum iteration
T=/nfsdata/DATASET/${dname}/${dname}_0.labels

GB=0.9                              # global beta
LB=0.8                              # local  beta
b=1                                 # threshold of #bucket IDs in a bin
s=3                                 # seeding algorithm (0-3, 3 is silk)

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 1 GPU
# ------------------------------------------------------------------------------
n=19928                             # number of data points
P=/nfsdata/DATASET/${dname}/1/${dname} # prefix for data set
O=results/${dname}/1/               # output folder

for m1 in 12 # 16 20 24 28
do 
  for h1 in 2
  do 
    for t in 2 # 3 4 5 
    do 
      for m2 in 8 # 12 16 20
      do 
        for h2 in 2 # 4
        do 
          for delta in 2 # 3 4 5 
          do 
            mpirun -n 1 -hostfile hosts ./silk -alg 0 -n ${n} -d ${d} \
              -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} -b ${b} -D ${delta} \
              -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
          done
        done
      done
    done
  done
done
