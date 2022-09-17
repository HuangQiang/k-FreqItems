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

GB=0.9                              # global beta
LB=0.8                              # local  beta
b=1                                 # threshold of #bucket IDs in a bin
s=3                                 # seeding algorithm (0-3, 3 is silk)

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 4 GPUs
# ------------------------------------------------------------------------------
n=250000000                         # number of data points
P=/nfsdata/DATASET/Criteo1B/4/${dname} # prefix for data set
O=results/${dname}/4/               # output folder

for m1 in 8 12 16
do 
  for h1 in 4 6 
  do 
    for t in 2000 3000 4000 
    do 
      for m2 in 6 8 10 
      do 
        for h2 in 6 
        do 
          for delta in 1000 2000 
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


for m1 in 8 12 16
do 
  for h1 in 4 
  do 
    for t in 2000 3000 4000 
    do 
      for m2 in 6 8 10 
      do 
        for h2 in 4
        do 
          for delta in 1000 
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
