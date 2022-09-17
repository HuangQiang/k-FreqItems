#!/bin/bash
make clean
make -j

# ------------------------------------------------------------------------------
#  basic parameters
# ------------------------------------------------------------------------------
dname=Avazu                         # data set name
d=1000000                           # dimension
F=int32                             # data format: uint8, uint16, int32, float32
GA=0.4                              # global alpha
LA=0.4                              # local  alpha
max_iter=3                          # maximum iteration

GB=0.9                              # global beta
LB=0.8                              # local  beta
b=1                                 # threshold of #bucket IDs in a bin
s=3                                 # seeding algorithm (0-3, 3 is silk)

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 4 GPUs
# ------------------------------------------------------------------------------
n=10107240                          # number of data points
P=/nfsdata/DATASET/Avazu/4/${dname} # prefix for data set
O=results/${dname}/4/               # output folder
t=300
delta=300

# ------------------------------------------------------------------------------
m1_list=(8 16 24)
h1=4
m2=6
h2=6
k_list=(1000 4500 7000)

length=`expr ${#k_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  k=${k_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# ------------------------------------------------------------------------------
m1_list=(8 16 24)
h1=4
m2=8
h2=6
k_list=(1300 5000 8000)

length=`expr ${#k_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  k=${k_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# ------------------------------------------------------------------------------
m1_list=(8 16 24)
h1=4
m2=10
h2=6
k_list=(1500 5500 8400)

length=`expr ${#k_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  k=${k_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# ------------------------------------------------------------------------------
m1=8
h1=4
m2_list=(6 8 10)
h2=4
k_list=(2800 3900 4200)

length=`expr ${#k_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m2=${m2_list[j]}
  k=${k_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# ------------------------------------------------------------------------------
m1=16
h1=4
m2_list=(6 8 10)
h2=4
k_list=(7000 7800 8300)

length=`expr ${#k_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m2=${m2_list[j]}
  k=${k_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# ------------------------------------------------------------------------------
m1=24
h1=4
m2_list=(6 8 10)
h2=4
k_list=(9500 9800 10000)

length=`expr ${#k_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m2=${m2_list[j]}
  k=${k_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# ------------------------------------------------------------------------------
m1=8
h1=6
m2_list=(6 8 10)
h2=4
k_list=(1400 2000 2500)

length=`expr ${#k_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m2=${m2_list[j]}
  k=${k_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# ------------------------------------------------------------------------------
m1=16
h1=6
m2_list=(6 8 10)
h2=4
k_list=(4900 6000 6500)

length=`expr ${#k_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m2=${m2_list[j]}
  k=${k_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# ------------------------------------------------------------------------------
m1=24
h1=6
m2_list=(6 8 10)
h2=4
k_list=(7500 8500 9000)

length=`expr ${#k_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m2=${m2_list[j]}
  k=${k_list[j]}
  
  mpirun -n 4 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done
