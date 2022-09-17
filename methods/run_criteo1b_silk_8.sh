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
max_iter=8                          # maximum iteration

GB=0.9                              # global beta
LB=0.8                              # local  beta
b=1                                 # threshold of #bucket IDs in a bin
s=3                                 # seeding algorithm (0-3, 3 is silk)

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 8 GPUs
# ------------------------------------------------------------------------------
n=125000000                         # number of data points
P=/nfsdata/DATASET/Criteo1B/8/${dname} # prefix for data set
O=results/${dname}/8/               # output folder

# ------------------------------------------------------------------------------
k=1000

m1=8
h1=4
t=3000
m2=10
h2=4
delta=2000

mpirun -n 8 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
  -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
  -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
  -F ${F} -P ${P} -O ${O}
sleep 1m 30s

# ------------------------------------------------------------------------------
k=2000

m1=8
h1=4
t=2000
m2=8
h2=4
delta=1000

mpirun -n 8 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
  -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
  -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
  -F ${F} -P ${P} -O ${O}
sleep 1m 30s

# ------------------------------------------------------------------------------
k=4000

m1=16
h1=4
t=3000
m2=6
h2=4
delta=1000

mpirun -n 8 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
  -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
  -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
  -F ${F} -P ${P} -O ${O}
sleep 1m 30s

# ------------------------------------------------------------------------------
k=6000

m1=16
h1=4
t=3000
m2=8
h2=4
delta=1000

mpirun -n 8 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
  -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
  -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
  -F ${F} -P ${P} -O ${O}
sleep 1m 30s

# ------------------------------------------------------------------------------
k=8000

m1=8
h1=4
t=1000
m2=8
h2=4
delta=1000
  
mpirun -n 8 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
  -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
  -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
  -F ${F} -P ${P} -O ${O}
sleep 1m 30s

# ------------------------------------------------------------------------------
k=10000

m1=16
h1=4
t=2000
m2=8
h2=4
delta=2000

mpirun -n 8 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
  -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
  -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
  -F ${F} -P ${P} -O ${O}
sleep 1m 30s
