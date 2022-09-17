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

# ------------------------------------------------------------------------------
k=1000

m1_list=(8 12 12 8 12 8)
h1_list=(4 4 6 4 4 4)
t_list=(2000 4000 2000 2000 4000 3000)
m2_list=(6 6 8 6 6 10)
h2_list=(4 4 4 4 4 4)
delta_list=(2000 2000 2000 1000 1000 2000)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  sleep 1m 30s
done

# ------------------------------------------------------------------------------
k=2000

m1_list=(8 12 12 16 16 12)
h1_list=(4 4 4 6 6 4)
t_list=(2000 3000 3000 2000 2000 4000)
m2_list=(8 6 6 6 8 10)
h2_list=(4 4 4 4 4 4)
delta_list=(1000 2000 1000 1000 2000 2000)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  sleep 1m 30s
done

# ------------------------------------------------------------------------------
k=4000

m1_list=(12 16 16 12 16 16)
h1_list=(4 4 4 4 4 4)
t_list=(2000 4000 3000 2000 3000 4000)
m2_list=(6 8 6 6 6 8)
h2_list=(4 4 4 4 4 4)
delta_list=(2000 2000 2000 1000 1000 1000)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  sleep 1m 30s
done

# ------------------------------------------------------------------------------
k=6000

m1_list=(16 12 12 16)
h1_list=(4 4 4 4)
t_list=(3000 2000 2000 3000)
m2_list=(8 8 10 8)
h2_list=(4 4 4 4)
delta_list=(2000 1000 2000 1000)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  sleep 1m 30s
done

# ------------------------------------------------------------------------------
k=8000

m1_list=(8 16 16)
h1_list=(4 4 4 )
t_list=(1000 3000 2000)
m2_list=(8 10 6)
h2_list=(4 4 4)
delta_list=(1000 2000 2000)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  sleep 1m 30s
done

# ------------------------------------------------------------------------------
k=10000

m1_list=(16 16)
h1_list=(4 4)
t_list=(2000 2000)
m2_list=(8 6)
h2_list=(4 4)
delta_list=(2000 1000)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  sleep 1m 30s
done
