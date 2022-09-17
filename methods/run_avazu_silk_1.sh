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
max_iter=10                         # maximum iteration

GB=0.9                              # global beta
LB=0.8                              # local  beta
b=1                                 # threshold of #bucket IDs in a bin
s=3                                 # seeding algorithm (0-3, 3 is silk)

# ------------------------------------------------------------------------------
#  k-freqitems: run on headnode with 1 GPU
# ------------------------------------------------------------------------------
n=40428960                          # number of data points
P=/nfsdata/DATASET/Avazu/1/${dname} # prefix for data set
O=results/${dname}/1/               # output folder

# ------------------------------------------------------------------------------
k=1000

m1_list=(8)      # (8	8	8	8	8	8	8	8	8	8	16	8	16	8	8	8	8	8	16	8	16	8	8	8)
h1_list=(6)      # (4	4	6	6	6	4	4	6	6	6	6	6	6	6	6	6	6	4	6	6	6	4	4	4)
t_list=(500)     # (500	500	500	300	500	500	500	300	300	500	500	500	500	300	500	300	500	300	500	300	500	500	300	500)
m2_list=(6)      # (8	8	6	6	6	10	10	8	6	8	6	8	6	10	10	8	10	6	8	10	8	6	6	6)
h2_list=(4)      # (6	6	4	6	4	6	6	6	6	4	6	4	6	6	4	6	4	6	6	6	6	4	6	4)
delta_list=(100) # (300	100	300	300	100	300	100	300	100	300	300	100	100	300	300	100	100	300	300	100	100	300	100	100)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# ------------------------------------------------------------------------------
k=2000

m1_list=(8)      # (8	8	8	16	24	16	8	24	8	16	8	16	16	16	24	8	24	16)
h1_list=(4)      # (4	4	4	4	6	4	6	6	6	6	4	6	4	4	6	6	6	6)
t_list=(500)     # (300	500	500	500	500	500	300	500	300	300	300	500	500	500	500	300	500	500)
m2_list=(10)     # (10	10	10	6	6	6	10	6	6	6	6	6	8	8	8	8	8	6)
h2_list=(4)      # (6	4	4	6	6	6	4	6	4	6	4	4	6	6	6	4	6	4)
delta_list=(300) # (100	300	100	300	300	100	300	100	100	300	300	300	300	100	300	100	100	100)

length=`expr ${#t_list[*]} - 1`
for j in $(seq 0 ${length})
do
  m1=${m1_list[j]}
  h1=${h1_list[j]}
  t=${t_list[j]}
  m2=${m2_list[j]}
  h2=${h2_list[j]}
  delta=${delta_list[j]}
  
  mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
    -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
    -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
    -F ${F} -P ${P} -O ${O}
done

# # ------------------------------------------------------------------------------
# k=3000

# m1_list=(16)
# h1_list=(4)
# t_list=(500)
# m2_list=(6)
# h2_list=(4)
# delta_list=(300)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
# done

# # ------------------------------------------------------------------------------
# k=4000

# m1_list=(16)
# h1_list=(4)
# t_list=(500)
# m2_list=(8)
# h2_list=(4)
# delta_list=(300)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
# done

# # ------------------------------------------------------------------------------
# k=5000

# m1_list=(16)
# h1_list=(4)
# t_list=(500)
# m2_list=(10)
# h2_list=(4)
# delta_list=(100)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
# done

# # ------------------------------------------------------------------------------
# k=6000

# m1_list=(24)
# h1_list=(4)
# t_list=(500)
# m2_list=(6)
# h2_list=(4)
# delta_list=(300)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
# done

# # ------------------------------------------------------------------------------
# k=7000

# m1_list=(24)
# h1_list=(4)
# t_list=(500)
# m2_list=(8)
# h2_list=(4)
# delta_list=(100)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
# done

# # ------------------------------------------------------------------------------
# k=8000

# m1_list=(24)
# h1_list=(4)
# t_list=(500)
# m2_list=(10)
# h2_list=(4)
# delta_list=(100)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
# done

# # ------------------------------------------------------------------------------
# k=9000

# m1_list=(24)
# h1_list=(4)
# t_list=(300)
# m2_list=(6)
# h2_list=(4)
# delta_list=(100)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
# done

# # ------------------------------------------------------------------------------
# k=10000

# m1_list=(24)
# h1_list=(4)
# t_list=(300)
# m2_list=(10)
# h2_list=(4)
# delta_list=(300)

# length=`expr ${#t_list[*]} - 1`
# for j in $(seq 0 ${length})
# do
#   m1=${m1_list[j]}
#   h1=${h1_list[j]}
#   t=${t_list[j]}
#   m2=${m2_list[j]}
#   h2=${h2_list[j]}
#   delta=${delta_list[j]}
  
#   mpirun -n 1 -hostfile hosts ./silk -alg 1 -n ${n} -d ${d} -k ${k} \
#     -m ${max_iter} -s ${s} -m1 ${m1} -h1 ${h1} -t ${t} -m2 ${m2} -h2 ${h2} \
#     -b ${b} -D ${delta} -GB ${GB} -LB ${LB} -GA ${GA} -LA ${LA} \
#     -F ${F} -P ${P} -O ${O}
# done
