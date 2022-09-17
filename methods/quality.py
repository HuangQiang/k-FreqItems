import numpy as np
import sys
import os

from os.path import isdir, isfile, join
from os      import makedirs
from time    import time
from sklearn import metrics


# ------------------------------------------------------------------------------
def read_labels(truth_addr, label_addr):
    
    with open(truth_addr, "rb") as f:
        truth = np.fromfile(f, np.int32)
        
    with open(label_addr, "rb") as f:
        label = np.fromfile(f, np.int32)
    
    return truth, label


# ------------------------------------------------------------------------------
def calc_purity(truth, label):
    
    (n,) = label.shape
    inv_list = dict()
    for i in range(n):
        if (label[i] not in inv_list):
            inv_list[label[i]] = list()
        inv_list[label[i]].append(truth[i])
    
    sum_v = 0
    for _,l in inv_list.items():
        max_f = np.amax(np.bincount(l))
        sum_v = sum_v + max_f
    
    return sum_v / n


# ------------------------------------------------------------------------------
def evaluate(label_fold, s, k, truth, label):
    
    start_time = time()
    
    fname = label_fold + "quality.csv"
    with open(fname, 'a+', newline='') as f:
        # calc metrics
        purity = calc_purity(truth, label)
        ari    = metrics.adjusted_rand_score(truth, label)
        nmi    = metrics.normalized_mutual_info_score(truth, label)
        ami    = metrics.adjusted_mutual_info_score(truth, label)
        homo   = metrics.homogeneity_score(truth, label)
        cmpl   = metrics.completeness_score(truth, label)
        v_m    = metrics.v_measure_score(truth, label)
        
        running_time = time() - start_time
        print("s=%d, k=%d, purity=%.6f, ari=%.6f, nmi=%.6f, time=%.6f" % (s, 
            k, purity, ari, nmi, running_time))
        
        # write results to disk
        f.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n" % (k, purity, 
            ari, nmi, ami, homo, cmpl, v_m, s))
        f.close()


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    
    start_time = time()
    n = int(sys.argv[1])      # number of labels
    k = int(sys.argv[2])      # number of classes 
    s = int(sys.argv[3])      # seeding algorithm
    truth_addr = sys.argv[4]  # ground truth label address
    label_fold = sys.argv[5]  # input label folder
    
    if s == 3:
        m1    = int(sys.argv[6])
        h1    = int(sys.argv[7])
        t     = int(sys.argv[8])
        m2    = int(sys.argv[9])
        h2    = int(sys.argv[10])
        delta = int(sys.argv[11])
        label_addr = label_fold + "%d_silk_%d_%d_%d_%d_%d_%d_0.labels" % (k, 
            m1, h1, t, m2, h2, delta)
    elif s == 0:
        label_addr = label_fold + "%d_random_0.labels" % k
    elif s == 1:
        label_addr = label_fold + "%d_kmeanspp_0.labels" % k
    elif s == 2:
        label_addr = label_fold + "%d_kmeansll_0.labels" % k
    else:
        exit()
    
    # read ground truth labels and input labels
    truth, label = read_labels(truth_addr, label_addr)

    # evaluate the quality: purity, ari, nmi
    evaluate(label_fold, s, k, truth, label)
    
    