#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  basic parameters
# ------------------------------------------------------------------------------
dname=KDD2012                       # data set name
d=54686452                          # dimension
F=int32                             # data format: uint8, uint16, int32, float32
max_iter=10                         # maximum iteration

# # ------------------------------------------------------------------------------
# #  determine global alpha & local alpha for k-freqitems on headnode with 4 GPUs
# # ------------------------------------------------------------------------------
# n=37409776                          # number of data points
# P=/nfsdata/DATASET/kdd12/4/${dname} # prefix for data set
# O=results/${dname}/4/               # output folder

# s=0
# for k in 10 100 1000
# do 
#   for GA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#   do 
#     for LA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#     do 
#       mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#         -m ${max_iter} -s ${s} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
#     done 
#   done 
# done

# derived parameters
GA=0.3                              # global alpha
LA=0.4                              # local  alpha

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 4 GPUs
# ------------------------------------------------------------------------------
n=37409776                          # number of data points
P=/nfsdata/DATASET/kdd12/4/${dname} # prefix for data set
O=results/${dname}/4/               # output folder

for s in 0 1 2
do 
  for k in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
  do 
    mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
      -m ${max_iter} -s ${s} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
  done
done

# # ------------------------------------------------------------------------------
# #  k-modes2: run on headnode with 4 GPUs
# # ------------------------------------------------------------------------------
# n=37409776                          # number of data points
# P=/nfsdata/DATASET/kdd12/4/${dname} # prefix for data set
# O=results/${dname}/4/               # output folder

# # derived parameters
# GA=1.0                              # global alpha
# LA=0.4                              # local  alpha

# for s in 1 2 # 0 1 2
# do 
#   for k in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
#   do 
#     mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#       -m ${max_iter} -s ${s} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
#   done
# done

# # ------------------------------------------------------------------------------
# #  k-modes1: run on headnode with 4 GPUs
# # ------------------------------------------------------------------------------
# n=37409776                          # number of data points
# P=/nfsdata/DATASET/kdd12/4/${dname} # prefix for data set
# O=results/${dname}/4/               # output folder

# # derived parameters
# GA=0.0                              # global alpha
# LA=0.0                              # local  alpha

# # *** new setting ***
# max_iter=1

# for s in 1 2 # 0 1 2
# do 
#   for k in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
#   do 
#     mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#       -m ${max_iter} -s ${s} -GA ${GA} -LA ${LA} -F ${F} -P ${P} -O ${O}
#   done
# done
