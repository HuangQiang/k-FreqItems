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


# ------------------------------------------------------------------------------
k=20

m1_list=(16 28 20 20 16 16 28 16 28 28 24 28 24 28 16 20 20 24 24 16 20 28 20 24 16 20 16 20 20 16 12 16 24 24)
h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
t_list=(4 4 2 3 3 5 4 2 4 2 2 3 3 5 4 2 3 4 2 3 5 4 4 3 4 5 3 3 2 2 3 5 2 3)
m2_list=(16 16 12 12 12 16 20 8 12 8 12 8 12 12 20 16 16 20 8 8 8 8 8 8 8 16 16 20 20 12 8 20 16 16)
h2_list=(2 4 4 4 4 2 4 2 4 2 2 2 2 2 2 4 4 4 4 2 2 2 2 4 2 2 4 4 4 2 2 2 2 2)
delta_list=(3 3 3 3 2 2 3 3 2 4 4 4 4 4 3 3 3 2 3 3 2 4 3 3 2 3 2 3 3 3 2 2 4 4)

# m1_list=(12)   # (20 16 16 16 20 20 16 20 16 12 12 16 20 12 20 16)
# h1_list=(2)    # (2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(4)     # (5 5 5 3 4 5 2 3 4 5 4 5 2 4 5 3)
# m2_list=(12)   # (8 16 12 8 16 16 8 16 8 16 8 12 16 12 8 12)
# h2_list=(4)    # (4 4 4 4 4 4 4 4 4 2 4 4 4 4 4 4)
# delta_list=(3) # (4 4 3 4 5 5 4 5 4 5 2 2 5 3 3 4)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done

# ------------------------------------------------------------------------------
k=40

m1_list=(20 16 12 24 24 16 28 16 28 20 12 28 16 12 20 12 24 24 12 28 12 28 24 12 20 20 20 16 28 24 28)
h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
t_list=(3 2 2 2 3 3 4 4 2 4 3 3 2 2 3 3 3 2 2 2 2 3 3 3 3 3 4 4 2 5 3)
m2_list=(8 16 8 12 12 16 20 12 8 12 12 8 20 12 12 16 20 20 16 12 20 12 8 20 8 16 8 20 16 20 16)
h2_list=(4 2 4 4 4 2 4 2 4 2 2 4 2 4 4 2 4 4 4 4 4 4 4 2 2 4 2 2 4 2 4)
delta_list=(2 3 2 3 3 3 2 2 3 3 2 3 3 2 2 2 3 3 2 3 2 3 2 2 3 2 2 2 3 3 3)

# m1_list=(16 16 20 12 20 20 20 12 12 20 12 16 20 20)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(4 4 3 3 5 4 2 4 2 4 3 5 5 5)
# m2_list=(8 12 12 16 16 12 12 16 8 8 8 8 8 16)
# h2_list=(4 4 4 2 4 4 4 2 4 2 4 2 2 4)
# delta_list=(2 3 4 4 3 4 4 4 3 5 3 4 5 2)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done

# ------------------------------------------------------------------------------
k=60

m1_list=(20 28 28 20 16 24 28 24 20 28 20 16 20 28 12 24 24 20 24 16 20 24 16 28)
h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
t_list=(3 3 2 3 3 3 3 2 2 5 3 2 4 4 2 3 3 2 4 3 3 4 2 5)
m2_list=(20 20 20 12 8 12 8 8 16 20 16 8 12 12 8 8 16 20 8 12 20 20 12 12)
h2_list=(4 4 4 2 2 4 4 2 2 2 2 4 2 2 2 2 4 2 2 2 2 2 4 2)
delta_list=(2 3 3 3 2 2 2 3 3 3 3 2 2 3 2 3 2 3 2 2 3 3 2 2)

# m1_list=(16 20 20 16 16 20 12 16 16 12 12 16 16)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(4 4 4 2 3 5 3 4 5 5 4 3 2)
# m2_list=(16 8 12 12 12 16 8 12 8 12 12 16 16)
# h2_list=(4 4 4 2 2 2 4 2 2 2 2 2 2)
# delta_list=(2 2 3 4 4 5 2 4 3 2 3 4 4)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done

# ------------------------------------------------------------------------------
k=80

m1_list=(24 12 28 24 28 20 28 16 16 12 16 24 28 20 12 16 28 24 28 24 20 24)
h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
t_list=(3 2 3 2 2 4 4 2 3 2 2 2 3 4 2 3 4 3 2 4 3 2)
m2_list=(20 12 12 12 8 16 16 16 16 16 20 16 16 20 20 20 8 16 12 12 8 20)
h2_list=(4 2 4 2 2 2 2 4 2 2 4 2 4 2 2 2 2 2 2 2 2 2)
delta_list=(2 2 2 3 3 2 3 2 2 2 2 3 2 2 2 2 2 3 3 2 2 3)

# m1_list=(12 20 20 16 12 16 12 12 16 16 20 20 12 16 16)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(2 2 3 2 4 3 3 3 5 5 4 4 2 3 2)
# m2_list=(8 8 8 12 8 12 8 16 8 16 8 16 12 16 16)
# h2_list=(2 2 2 4 2 4 2 4 2 2 2 4 2 4 4)
# delta_list=(3 4 4 3 2 3 3 2 2 3 4 2 3 3 3)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done

# ------------------------------------------------------------------------------
k=100

m1_list=(28 20 24 28 28 28 20 28 16 20 28 20 28 28)
h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2)
t_list=(3 2 3 3 5 2 2 3 2 3 4 2 2 3)
m2_list=(20 8 20 12 20 16 12 16 8 12 12 16 20 20)
h2_list=(4 4 2 2 2 2 4 2 2 2 2 4 2 2)
delta_list=(2 2 3 3 2 3 2 3 2 2 2 2 3 3)

# m1_list=(12 12 20 20 16 20 16 16 20 20 12 20)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(4 3 4 2 4 3 5 3 3 2 4 4)
# m2_list=(12 16 12 8 12 8 12 8 16 16 16 16)
# h2_list=(2 2 2 4 2 4 2 4 2 2 2 2)
# delta_list=(2 3 4 3 3 3 2 2 4 4 2 4)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done

# ------------------------------------------------------------------------------
k=120

m1_list=(20 24 24 16 24 20 28 16 24 20 16 28 28)
h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2)
t_list=(2 4 3 2 2 3 4 2 2 3 2 4 3)
m2_list=(20 20 8 12 8 16 16 16 12 20 20 20 8)
h2_list=(4 2 2 2 4 2 2 2 4 2 2 2 2)
delta_list=(2 2 2 2 2 2 2 2 2 2 2 2 2)

# m1_list=(16 20 20 16 20 16 16 20 20 16 20 20 16 20 16 16 12)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(4 2 5 5 3 3 2 2 3 3 5 4 4 5 3 2 3)
# m2_list=(16 12 12 16 12 12 8 16 16 8 8 8 8 16 16 12 8)
# h2_list=(2 4 2 2 4 4 2 4 4 2 2 2 2 2 4 2 2)
# delta_list=(3 3 3 2 3 2 3 3 3 3 2 3 2 3 2 3 2)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done

# ------------------------------------------------------------------------------
k=140

m1_list=(24 24 24 28 20 24 28 28 20 24)
h1_list=(2 2 2 2 2 2 2 2 2 2)
t_list=(2 3 2 2 2 3 2 3 2 3)
m2_list=(16 12 20 8 8 16 12 12 12 20)
h2_list=(4 2 4 4 2 2 4 2 2 2)
delta_list=(2 2 2 2 2 2 2 2 2 2)

# m1_list=(16 20 16 12 16 16 20 12 12 20)
# h1_list=(2 2 2 2 2 2 2 2 2 2)
# t_list=(3 3 2 2 3 4 4 3 2 5)
# m2_list=(12 8 16 8 16 12 12 12 12 12)
# h2_list=(2 4 2 4 2 2 2 2 4 2)
# delta_list=(3 2 3 2 3 2 3 2 2 2)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done

# ------------------------------------------------------------------------------
k=160

m1_list=(28 28 20 28 24 20 28)
h1_list=(2 2 2 2 2 2 2)
t_list=(2 2 2 3 2 2 3)
m2_list=(16 20 16 16 8 20 20)
h2_list=(4 4 2 2 2 2 2)
delta_list=(2 2 2 2 2 2 2)

# m1_list=(20 12 12 20 16 20 20 20 20 20)
# h1_list=(2 2 2 2 2 2 2 2 2 2)
# t_list=(3 3 2 4 4 2 3 5 3 4)
# m2_list=(12 16 16 16 16 8 8 16 16 8)
# h2_list=(4 2 4 2 2 2 2 2 4 2)
# delta_list=(2 2 2 3 2 3 3 2 2 2)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done

# ------------------------------------------------------------------------------
k=180

m1_list=(24 28 24 24 28)
h1_list=(2 2 2 2 2)
t_list=(2 2 2 2 2)
m2_list=(12 8 16 20 12)
h2_list=(2 2 2 2 2)
delta_list=(2 2 2 2 2)

# m1_list=(20 20 16 20 20 16 20 12 16 16 12 20)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(2 3 3 2 3 2 4 2 3 2 2 4)
# m2_list=(12 12 8 16 16 8 12 8 12 12 12 16)
# h2_list=(2 2 2 2 2 4 2 2 2 4 2 2)
# delta_list=(3 3 2 3 3 2 2 2 2 2 2 2)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done

# ------------------------------------------------------------------------------
k=200

m1_list=(28 28)
h1_list=(2 2)
t_list=(2 2)
m2_list=(16 20)
h2_list=(2 2)
delta_list=(2 2)

# m1_list=(16 16 12 20 20 20 16 20 20 16 20 16 20 20 20)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(2 3 2 3 2 2 2 3 2 2 3 2 2 2 2)
# m2_list=(16 16 16 8 8 12 8 12 16 12 16 16 8 12 16)
# h2_list=(4 2 2 2 4 4 2 2 4 2 2 2 2 2 2)
# delta_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
  
  python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
done






# # ------------------------------------------------------------------------------
# k=50

# m1_list=(16 28 20 28 20 16 28 16 12 16 28 16 16 12 24 28 24 12 16 16 20 20 20 20 16 24 24 24 28 16)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(4 3 4 2 4 2 4 3 3 4 5 4 5 5 5 5 3 4 3 2 2 5 5 4 4 4 3 4 3 2)
# m2_list=(16 8 8 8 12 12 8 12 8 20 20 12 8 12 20 12 16 12 16 16 20 20 8 16 16 8 20 20 12 8)
# h2_list=(4 4 4 4 4 2 4 2 4 4 4 2 2 2 4 4 4 2 2 2 2 2 2 4 2 4 4 4 4 4)
# delta_list=(2 4 2 4 3 4 4 4 2 2 4 4 3 2 3 3 4 3 4 4 5 5 4 3 4 3 4 4 4 3)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
  
#   python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
# done

# # ------------------------------------------------------------------------------
# k=100

# m1_list=(12 24 24 16 28 16 28 20 12 28 16 12 24 20 24 16 28 28 20)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(2 2 3 3 4 4 2 4 3 3 2 2 2 5 3 3 2 3 3)
# m2_list=(8 12 12 16 20 12 8 12 12 8 20 12 16 12 16 20 16 16 12)
# h2_list=(4 4 4 2 4 2 4 2 2 4 2 4 4 2 4 2 2 2 4)
# delta_list=(2 3 3 3 2 2 3 3 2 3 3 2 3 2 3 3 4 4 2)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
  
#   python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
# done

# # ------------------------------------------------------------------------------
# k=200

# m1_list=(16 12 24 24 16 20 16 24 12 28 24 28 20 28 16 24 16 12 24 16 28 24 28 28 20 12)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(2 2 3 4 3 3 2 3 2 3 2 2 4 4 2 3 3 2 5 2 3 2 3 4 4 2)
# m2_list=(8 8 16 8 12 20 12 20 12 12 12 8 16 16 16 12 16 16 20 20 8 16 16 20 20 20)
# h2_list=(4 2 4 2 2 2 4 4 2 4 2 2 2 2 4 2 2 2 2 4 2 2 4 2 2 2)
# delta_list=(2 2 2 2 2 3 2 2 2 2 3 3 2 3 2 3 2 2 2 2 3 3 2 3 2 2)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
  
#   python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
# done

# # ------------------------------------------------------------------------------
# k=300

# m1_list=(20 16 20 28 20 24 24 16 24 20 28 16 24 20)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(2 2 2 3 2 4 3 2 2 3 4 2 2 3)
# m2_list=(12 8 16 20 20 20 8 12 8 16 16 16 12 20)
# h2_list=(4 2 4 2 4 2 2 2 4 2 2 2 4 2)
# delta_list=(2 2 2 3 2 2 2 2 2 2 2 2 2 2)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
  
#   python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
# done

# # ------------------------------------------------------------------------------
# k=400

# m1_list=(24 28 28 28 20 24 28 28 20 28 24 20)
# h1_list=(2 2 2 2 2 2 2 2 2 2 2 2)
# t_list=(2 2 2 3 2 3 2 2 2 3 2 2)
# m2_list=(20 8 12 12 12 20 16 20 16 16 8 20)
# h2_list=(4 4 4 2 2 2 4 4 2 2 2 2)
# delta_list=(2 2 2 2 2 2 2 2 2 2 2 2)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
  
#   python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
# done

# # ------------------------------------------------------------------------------
# k=500

# m1_list=(28 24 28 24 24 28 28 28)
# h1_list=(2 2 2 2 2 2 2 2)
# t_list=(3 2 2 2 2 2 2 2)
# m2_list=(20 12 8 16 20 12 16 20)
# h2_list=(2 2 2 2 2 2 2 2)
# delta_list=(2 2 2 2 2 2 2 2)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   CUDA_VISIBLE_DEVICES=2 mpirun -bind-to none --cpu-set 2-51 -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
  
#   python quality.py ${n} ${k} ${s} ${T} ${O} ${m1} ${h1} ${t} ${m2} ${h2} ${delta}
# done
