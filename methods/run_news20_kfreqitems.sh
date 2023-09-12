#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  basic parameters
# ------------------------------------------------------------------------------
dname=News20                        # data set name
d=62061                             # dimension
F=int32                             # data format: uint8, uint16, int32, float32
max_iter=10                         # maximum iteration
T=/nfsdata/DATASET/${dname}/${dname}_0.labels

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 1 GPU
# ------------------------------------------------------------------------------
n=19928                             # number of data points
P=/nfsdata/DATASET/${dname}/1/${dname} # prefix for data set
O=results/${dname}/1/               # output folder

# derived parameters
GA=0.2                              # global alpha
LA=0.2                              # local  alpha

for s in 0 1 2
do
  for k in 20 40 60 80 100 120 140 160 180 200
  do
    CUDA_VISIBLE_DEVICES=1 mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
      -m ${max_iter} -s ${s} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
    
    python quality.py ${n} ${k} ${s} ${T} ${O}
  done
done

# # ------------------------------------------------------------------------------
# #  k-modes2: run on headnode with 1 GPU
# # ------------------------------------------------------------------------------
# n=19928                             # number of data points
# P=/nfsdata/DATASET/${dname}/1/${dname} # prefix for data set
# O=results/${dname}/1/               # output folder

# # derived parameters
# GA=1.0                              # global alpha
# LA=0.2                              # local  alpha

# for s in 0
# do
#   for k in 20 40 60 80 100 120 140 160 180 200
#   do
#     mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#       -m ${max_iter} -s ${s} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
#   done
# done

# # ------------------------------------------------------------------------------
# #  k-modes1: run on headnode with 1 GPU
# # ------------------------------------------------------------------------------
# n=19928                             # number of data points
# P=/nfsdata/DATASET/${dname}/1/${dname} # prefix for data set
# O=results/${dname}/1/               # output folder

# # derived parameters
# GA=0.0                              # global alpha
# LA=0.0                              # local  alpha

# for s in 0
# do
#   for k in 20 40 60 80 100 120 140 160 180 200
#   do
#     mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#       -m ${max_iter} -s ${s} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
#   done
# done
