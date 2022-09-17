#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  basic parameters
# ------------------------------------------------------------------------------
dname=KDD2012                       # data set name
d=54686452                          # dimension
F=int32                             # data format: uint8, uint16, int32, float32
GA=0.3                              # global alpha
LA=0.4                              # local  alpha
max_iter=10                         # maximum iteration

GB=0.9                              # global beta
LB=0.8                              # local  beta
b=1                                 # threshold of #bucket IDs in a bin
s=3                                 # seeding algorithm (0-3, 3 is silk)

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 4 GPUs
# ------------------------------------------------------------------------------
n=37409776                          # number of data points
P=/nfsdata/DATASET/kdd12/4/${dname} # prefix for data set
O=results/${dname}/4/               # output folder

# ------------------------------------------------------------------------------
k=1000

m1_list=(8)       # (8 8 8 8 8 8 8 8 8 8 12 8 8 8 8 12 8 8 8 8 8 8 12 8 8 12 8 12 8 8 8 12 8 8 8 8 8 8)
h1_list=(8)       # (8 8 8 8 8 8 8 6 8 6 8 8 6 8 6 8 8 6 8 8 6 8 8 8 8 8 8 8 8 6 4 8 4 6 8 6 8 6)
t_list=(3000)     # (3000 3000 3000 3000 3000 3000 2000 3000 1000 3000 3000 2000 3000 3000 3000 3000 1000 3000 3000 2000 3000 1000 3000 3000 2000 3000 3000 3000 3000 2000 3000 3000 3000 1000 2000 3000 2000 3000)
m2_list=(6)       # (6 6 8 8 10 10 6 6 6 6 6 8 8 6 8 6 8 10 6 10 10 10 8 8 6 8 8 10 10 6 6 10 6 6 6 6 8 6)
h2_list=(4)       # (6 6 6 6 6 6 6 6 6 6 6 6 6 4 6 6 6 6 4 6 6 6 6 4 6 6 4 6 4 6 6 6 6 6 4 4 6 4)
delta_list=(1000) # (2000 1000 2000 1000 2000 1000 2000 2000 2000 1000 2000 2000 2000 2000 1000 1000 2000 2000 1000 2000 1000 2000 2000 2000 1000 1000 1000 2000 2000 2000 2000 1000 1000 2000 2000 2000 1000 1000)

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
done

# ------------------------------------------------------------------------------
k=2000

m1_list=(12)      # (8 12 8 12 12 8 8 8 12 8 8 8 8 8 8 8 12 8 8 12 12 8 12 8 8 8 12 8 12 12 12 8 12 12 12 8 8 8 8 8 8 12 8 12 12 12 8 8 12 12 12 8 8 12 8 12 8 8)
h1_list=(8)       # (8 8 8 6 8 4 4 6 6 6 8 4 6 4 8 6 8 6 8 6 8 6 6 6 8 6 8 8 8 6 8 6 6 8 8 4 4 4 4 6 6 8 6 8 8 6 8 6 8 6 8 4 4 8 4 6 4 6)
t_list=(3000)     # (3000 2000 1000 3000 1000 3000 3000 2000 3000 1000 2000 3000 2000 3000 2000 3000 3000 1000 1000 3000 2000 3000 3000 2000 2000 3000 1000 1000 2000 3000 3000 3000 3000 3000 1000 2000 3000 1000 3000 2000 2000 3000 1000 3000 2000 2000 2000 2000 3000 1000 2000 2000 3000 1000 1000 3000 3000 2000)
m2_list=(6)       # (10 6 6 6 6 8 8 8 6 8 10 10 10 10 8 8 6 10 8 8 8 8 8 6 10 10 8 10 10 10 6 10 10 8 10 6 6 6 6 6 8 10 6 8 6 6 6 10 10 6 6 8 8 6 8 6 8 8)
h2_list=(4)       # (4 6 4 6 6 6 6 6 6 6 6 6 6 6 4 4 4 6 4 6 6 4 6 6 4 4 6 4 6 6 4 4 6 4 6 6 4 6 4 4 6 4 4 4 6 6 4 6 4 6 4 6 4 4 6 4 4 4)
delta_list=(2000) # (1000 2000 2000 2000 2000 1000 2000 2000 1000 2000 1000 2000 2000 1000 2000 2000 2000 2000 2000 2000 2000 1000 1000 1000 2000 2000 2000 2000 2000 2000 1000 1000 1000 2000 2000 2000 2000 2000 1000 2000 1000 2000 2000 1000 1000 2000 1000 1000 1000 2000 2000 2000 2000 2000 2000 2000 1000 2000)

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
done

# ------------------------------------------------------------------------------
k=3000

m1_list=(12)      # (8 12 8 12 8 8 12 8 12 8 8 8 8 12 12 8 8 8 8)
h1_list=(6)       # (4 8 4 6 4 6 4 6 4 4 4 6 6 6 8 8 4 6 4)
t_list=(3000)     # (3000 1000 1000 3000 3000 2000 3000 1000 3000 3000 3000 2000 1000 3000 2000 2000 2000 2000 1000)
m2_list=(6)       # (8 6 8 6 8 8 6 8 6 10 10 10 10 6 8 10 6 6 6)
h2_list=(4)       # (4 4 6 4 4 4 6 4 6 4 4 4 4 4 6 4 4 4 4)
delta_list=(2000) # (2000 2000 2000 2000 1000 2000 2000 2000 1000 2000 1000 2000 2000 1000 1000 1000 2000 1000 2000)

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
done

# ------------------------------------------------------------------------------
k=4000

m1_list=(12)      # (12 12 8 12 8 8 8 8 8 12 8 12 12 8 12)
h1_list=(6)       # (4 6 4 6 8 4 6 8 4 4 6 6 6 4 4)
t_list=(1000)     # (3000 2000 2000 3000 1000 2000 2000 1000 2000 3000 2000 2000 1000 1000 3000)
m2_list=(6)       # (10 6 10 10 8 8 8 10 10 6 10 6 6 8 6)
h2_list=(4)       # (6 6 6 4 6 4 4 6 4 4 4 4 4 4 4)
delta_list=(2000) # (1000 1000 1000 1000 1000 2000 1000 1000 2000 1000 1000 2000 2000 2000 2000)

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
done


# ------------------------------------------------------------------------------
k=5000

m1_list=(8)       # (12 12 8 8)
h1_list=(6)       # (4 4 6 6)
t_list=(1000)     # (2000 2000 1000 1000)
m2_list=(8)       # (10 6 8 10)
h2_list=(4)       # (4 4 4 4)
delta_list=(1000) # (2000 1000 1000 1000)

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
done

# ------------------------------------------------------------------------------
k=6000

m1_list=(12)      # (12 12)
h1_list=(8)       # (8 4)
t_list=(1000)     # (1000 2000)
m2_list=(6)       # (6 8)
h2_list=(4)       # (4 4)
delta_list=(1000) # (1000 1000)

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
done

# ------------------------------------------------------------------------------
k=7000

m1_list=(12)      # (12 12)
h1_list=(8)       # (4 8)
t_list=(1000)     # (2000 1000)
m2_list=(10)      # (10 10)
h2_list=(4)       # (4 4)
delta_list=(1000) # (1000 1000)

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
done

# ------------------------------------------------------------------------------
k=8000

m1_list=(12)      # (12 12)
h1_list=(6)       # (8 6)
t_list=(1000)     # (1000 1000)
m2_list=(6)       # (10 6)
h2_list=(4)       # (4 4)
delta_list=(1000) # (1000 1000)

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
done

# ------------------------------------------------------------------------------
k=9000

m1_list=(12)      # (12 12)
h1_list=(6)       # (6 6)
t_list=(1000)     # (1000 1000)
m2_list=(8)       # (6 8)
h2_list=(4)       # (4 4)
delta_list=(1000) # (1000 1000)

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
done

# ------------------------------------------------------------------------------
k=10000

m1_list=(12)      # (12 12)
h1_list=(6)       # (6 6)
t_list=(1000)     # (1000 1000)
m2_list=(8)       # (8 10)
h2_list=(4)       # (4 4)
delta_list=(1000) # (1000 1000)

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
done
