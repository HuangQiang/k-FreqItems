import os
import sys
import inspect

import math
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from os        import makedirs
from os.path   import isdir, isfile, join
from plot_util import PlotHelper
from mpl_toolkits.mplot3d import Axes3D

markers = ['s','o','^','x','D','v','>','o','v','^','s','D','x','>']
colors  = ['crimson', 'blueviolet', 'forestgreen', 'blue', 'orange',
           'olive', 'violet', 'skyblue', 'lawngreen', 'cyan', 'purple',  
           'red', 'deeppink', 'lime', 'glod']


# ------------------------------------------------------------------------------
def read_csv(file_name, x_name, y_name):
    '''
    read x,y results from csv file

    :params file_name: file name (string)
    :params x_name: the name of x-axis (string)
    :params y_name: the name of y-axis (string)
    :returns: x, y
    '''
    data = {}
    with open(file_name, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        data['k']     = []
        data['mse']   = []
        data['mae']   = []
        data['wtime'] = []
        data['ctime'] = []
        i = 0
        for row in reader:
            i = i + 1
            if i == 1: continue # skip first row
            
            data['k'].append(int(row[0]))
            data['mse'].append(float(row[1]))
            data['mae'].append(float(row[2]))
            data['wtime'].append(float(row[3]))
            data['ctime'].append(float(row[4]))
    # print(data)
    return data[x_name], data[y_name]


# ------------------------------------------------------------------------------
def read_csv2(file_name, x_name, y_name):
    '''
    read x,y results from csv file

    :params file_name: file name (string)
    :params x_name: the name of x-axis (string)
    :params y_name: the name of y-axis (string)
    :returns: x, y
    '''
    data = {}
    with open(file_name, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        data['k']     = []
        data['mse']   = []
        data['mae']   = []
        data['wtime'] = []
        data['ctime'] = []
        data['clust'] = []
        data['total'] = []
        data['iter']  = []
        i = 0
        for row in reader:
            i = i + 1
            if i > 10: break
            
            data['k'].append(int(row[0]))
            data['mse'].append(float(row[1]))
            data['mae'].append(float(row[2]))
            data['wtime'].append(float(row[3]))
            data['ctime'].append(float(row[4]))
            data['clust'].append(int(row[5]))
            data['total'].append(int(row[6]))
            data['iter'].append(int(row[7]))
    # print(data)
    return data[x_name], data[y_name]
    

# ------------------------------------------------------------------------------
def read_csv_alpha(file_name, x_name, y_name, z_name):
    '''
    read x,y results from csv file

    :params file_name: file name (string)
    :params x_name: the name of x-axis (string)
    :params y_name: the name of y-axis (string)
    :returns: x, y
    '''
    data = {}
    with open(file_name, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        data['k']        = []
        data['mse']      = []
        data['mae']      = []
        data['wtime']    = []
        data['ctime']    = []
        data['clust']    = []
        data['total']    = []
        data['iter']     = []
        data['g_alpha']  = []
        data['l_alpha']  = []
        i = 0
        for row in reader:
            i = i + 1
            if i == 1: continue # skip first row
            
            data['k'].append(int(row[0]))
            data['mse'].append(float(row[1]))
            data['mae'].append(float(row[2]))
            data['wtime'].append(float(row[3]))
            data['ctime'].append(float(row[4]))
            data['clust'].append(int(row[5]))
            data['total'].append(int(row[6]))
            data['iter'].append(int(row[7]))
            data['g_alpha'].append(float(row[8]))
            data['l_alpha'].append(float(row[9]))
    # print(data)
    return data[x_name], data[y_name], data[z_name]
    

# ------------------------------------------------------------------------------
def read_csv_truth(file_name, x_name, y_name):
    '''
    read x,y results from csv file

    :params file_name: file name (string)
    :params x_name: the name of x-axis (string)
    :params y_name: the name of y-axis (string)
    :returns: x, y
    '''
    data = {}
    with open(file_name, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        data['k']      = []
        data['ari']    = []
        data['mse']    = []
        data['mae']    = []
        data['purity'] = []
        i = 0
        for row in reader:
            i = i + 1
            if i == 1: continue # skip first row
            
            data['k'].append(int(row[0]))
            data['ari'].append(float(row[1]))
            data['mse'].append(float(row[2]))
            data['mae'].append(float(row[3]))
            data['purity'].append(float(row[4]))
    # print(data)
    return data[x_name], data[y_name]
    

# ------------------------------------------------------------------------------
def choice_alpha(file_name, dname_list, output_folder, xx, yy, zz_list, 
    fig_width, fig_height):
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    num_height = 1
    num_width  = len(dname_list)
    cnt  = 0
    for di, dname in enumerate(dname_list):
        ax = fig.add_subplot(num_height, num_width, di+1, projection='3d')
        
        x,y,z = read_csv_alpha('%s/choice_alpha.csv' % dname, 'g_alpha', 
            'l_alpha', 'mse')
        ax.scatter(x, y, z, c=colors[cnt], marker=markers[cnt], linewidths=2)
        cnt += 1
    
        ax.set_xlabel(r'Global $\alpha$', fontsize=12)
        ax.set_ylabel(r'Local $\alpha$', fontsize=12)
        ax.set_zlabel(r'MSE', fontsize=12)
        ax.zaxis.set_rotate_label(True) 
        
        
        zz = zz_list[di]
        ax.set_xticks(np.arange(xx[0], xx[1], step=xx[2]))
        ax.set_yticks(np.arange(yy[0], yy[1], step=yy[2]))
        ax.set_zticks(np.arange(zz[0], zz[1], step=zz[2]))
        # ax.invert_xaxis()
        
        ax.set_title('%s'%dname, fontsize=13, y=1.0) # 
        
    if not isdir(output_folder):
        makedirs(output_folder)
        
    filename = join(output_folder, file_name)
    plt.savefig('%s.png' % filename, format='png', dpi=1000, bbox_inches="tight")
    plt.savefig('%s.eps' % filename, format='eps', dpi=1000, bbox_inches="tight")
    plt.savefig('%s.pdf' % filename, format='pdf', dpi=1000, bbox_inches="tight")
    plt.show()


# ------------------------------------------------------------------------------
def seeding(file_name, dname_list, legend_list, output_folder, mname_list, 
    ylabel_list, xx, yy_matrix, fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    num_height = len(mname_list)
    num_width  = len(dname_list)
    
    for mi, mname, in enumerate(mname_list):
        for di, dname in enumerate(dname_list):
            ax = plt.subplot(num_height, num_width, mi*num_width+di+1)
            ax.set_xlabel(r'$k$')
            if (di == 0):
                ax.set_ylabel(r'%s' % ylabel_list[mi])
            if (mi == 0):
                ax.set_title('%s' % dname, fontsize=14)
            cnt  = 0
    
            # plot silk
            x,y = read_csv('%s/silk_1.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
        
            # plot k-means||
            x,y = read_csv('%s/kmeansll_1.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
            
            # plot k-means++
            x,y = read_csv('%s/kmeanspp_1.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
            
            # plot random
            x,y = read_csv('%s/random_1.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
            
            yy = yy_matrix[di][mi]
            plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
            plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        
    plt_helper.plot_fig_legend(ncol=4, font_size=15, legend_width=0.8,
        legend_start=0.99)
    plt_helper.plot_and_save(output_folder, file_name)


# ------------------------------------------------------------------------------
def clustering(file_name, dname_list, legend_list, output_folder, 
    mname_list, ylabel_list, xx, yy_matrix, fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    num_height = len(mname_list)
    num_width  = len(dname_list)
    
    for mi, mname, in enumerate(mname_list):
        for di, dname in enumerate(dname_list):
            ax = plt.subplot(num_height, num_width, mi*num_width+di+1)
            ax.set_xlabel(r'$k$')
            if (di == 0):
                ax.set_ylabel(r'%s' % ylabel_list[mi])
            if (mi == 0):
                ax.set_title('%s' % dname, fontsize=14)
            cnt  = 0

            # plot silk
            x,y = read_csv('%s/silk.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
        
            # plot k-means||
            x,y = read_csv('%s/kmeansll.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
            
            # plot k-means++
            x,y = read_csv('%s/kmeanspp.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
            
            # plot random
            x,y = read_csv('%s/random.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
            
            # plot svd + kmeans
            x,y = read_csv('%s/lsa_kmeans.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
            
            # plot se + kmeans
            x,y = read_csv('%s/se_kmeans.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="%s" % legend_list[cnt] if mi==0 and di==0 else "")
            cnt += 1
            
            yy = yy_matrix[di][mi]
            plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
            plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        
    plt_helper.plot_fig_legend(ncol=6, font_size=15, legend_width=0.8,
        legend_start=0.99)
    plt_helper.plot_and_save(output_folder, file_name)


# ------------------------------------------------------------------------------
def convergence(file_name, dname_list, k, output_folder, mname_list, 
    ylabel_list, xx, yy_matrix, fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    num_height = len(mname_list)
    num_width  = len(dname_list)
    
    for mi, mname in enumerate(mname_list):
        for di, dname in enumerate(dname_list):
            ax = plt.subplot(num_height, num_width, mi*num_width+di+1)
            ax.set_xlabel(r'$\#$ Iterations')
            if di == 0: ax.set_ylabel(r'%s' % ylabel_list[mi])
            if mi == 0: ax.set_title('%s' % dname, fontsize=14)
            cnt  = 0
    
            # plot silk
            x,y = read_csv2('%s/silk_%d.csv' % (dname,k), 'iter', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="SILK" if mi==0 and di==0 else "")
            cnt += 1
        
            # plot k-means||
            x,y = read_csv2('%s/kmeansll_%d.csv' % (dname,k), 'iter', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="$k$-FreqItems||" if mi==0 and di==0 else "")
            cnt += 1
            
            # plot k-means++
            x,y = read_csv2('%s/kmeanspp_%d.csv' % (dname,k), 'iter', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="$k$-FreqItems++" if mi==0 and di==0 else "")
            cnt += 1
            
            # plot random
            x,y = read_csv2('%s/random_%d.csv' % (dname,k), 'iter', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7,
                label="$k$-FreqItems" if mi==0 and di==0 else "")
            cnt += 1
    
            yy = yy_matrix[di][mi]
            plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
            plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        
    plt_helper.plot_fig_legend(ncol=4, font_size=15, legend_width=0.8,
        legend_start=0.99)
    file_name = file_name + "_%d" % k
    plt_helper.plot_and_save(output_folder, file_name)


# ------------------------------------------------------------------------------
def multi_gpus(file_name, legend_name, dname_list, iter_list, output_folder, 
        mname_list, ylabel_list, xx, yy_matrix, fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    num_rows = len(mname_list)
    num_cols = len(dname_list)
    
    for mi, mname in enumerate(mname_list):
        for di, dname in enumerate(dname_list):
            ax = plt.subplot(num_rows, num_cols, mi*num_cols+di+1)
            ax.set_xlabel(r'$k$')
            if di == 0: ax.set_ylabel(r'%s' % ylabel_list[mi])
            if mi == 0: ax.set_title('%s' % dname, fontsize=13)
        
            cnt  = 0
            for iter in iter_list:
                fname = '%s/%s_1_%d.csv' % (dname, file_name, iter)
                x,y = read_csv(fname, 'k', mname)
                if iter == 1:
                    ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7, 
                        label="%d GPU" % iter if mi==0 and di==0 else "")
                else:
                    ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7, 
                        label="%d GPUs" % iter if mi==0 and di==0 else "")
                # ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7, 
                #     label="%s (%d)" % (legend_name,iter) if mi==0 and di==0 else "")
                cnt += 1
            
            yy = yy_matrix[di][mi]
            plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
            plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    plt_helper.plot_fig_legend(ncol=4, font_size=13, legend_width=0.8, 
        legend_start=0.97)
    plt_helper.plot_and_save(output_folder, 'multi_gpus_%s' % file_name)


# ------------------------------------------------------------------------------
def billion(dname, iter_list, title_list, output_folder, mname_list, 
    ylabel_list, xx, yy_matrix, fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.8, hspace=0.37)
    
    num_rows = len(mname_list)
    num_cols = len(iter_list)
    
    for mi, mname in enumerate(mname_list):
        for ii, iter in enumerate(iter_list):
            ax = plt.subplot(num_rows, num_cols, mi*num_cols+ii+1)
            ax.set_xlabel(r'$k$')
            if ii == 0: ax.set_ylabel(r'%s' % ylabel_list[mi])
            if mi == 0: ax.set_title('%s' % title_list[ii], fontsize=13)
            cnt = 0
            
            # plot silk
            x,y = read_csv('%s/silk_%d.csv' % (dname,iter), 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7, 
                label="SILK" if mi==0 and ii==0 else "")
            cnt += 1
            
            # plot k-means||
            x,y = read_csv('%s/kmeansll_%d.csv' % (dname,iter), 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7, 
                label="$k$-FreqItems||" if mi==0 and ii==0 else "")
            cnt += 1
            
            # plot k-means++
            x,y = read_csv('%s/kmeanspp_%d.csv' % (dname,iter), 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7, 
                label="$k$-FreqItems++" if mi==0 and ii==0 else "")
            cnt += 1
            
            # plot random
            x,y = read_csv('%s/random_%d.csv' % (dname,iter), 'k', mname)
            ax.plot(x, y, color=colors[cnt], marker=markers[cnt], markersize=7, 
                label="$k$-FreqItems" if mi==0 and ii==0 else "")
            cnt += 1
        
            yy = yy_matrix[ii][mi]
            plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
            plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    plt_helper.plot_fig_legend(ncol=4, font_size=13, legend_width=0.8, 
        legend_start=0.98)
    plt_helper.plot_and_save(output_folder, "billion")


# ------------------------------------------------------------------------------
def truth_full(file_name, dname_list, legend_list, output_folder, mname_list, 
    ylabel_list, xx_list, yy_matrix, fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=1.2, bottom_space=0.01, hspace=0.37)
    
    num_height = len(mname_list)
    num_width  = len(dname_list)
    
    for mi, mname, in enumerate(mname_list):
        for di, dname in enumerate(dname_list):
            ax = plt.subplot(num_height, num_width, mi*num_width+di+1)
            ax.set_xlabel(r'$k$')
            if (di == 0):
                ax.set_ylabel(r'%s' % ylabel_list[mi])
            if (mi == 0):
                ax.set_title('%s' % dname, fontsize=14)
    
            # plot silk
            x,y = read_csv_truth('%s/silk.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[0], marker=markers[0], markersize=7,
                label="%s" % legend_list[0] if mi==0 and di==0 else "")
            
            # plot random
            x,y = read_csv_truth('%s/random.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[3], marker=markers[3], markersize=7,
                label="%s" % legend_list[3] if mi==0 and di==0 else "")
                
            # plot k-means||
            x,y = read_csv_truth('%s/kmeansll.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[1], marker=markers[1], markersize=7,
                label="%s" % legend_list[1] if mi==0 and di==0 else "")
            
            # plot k-means++
            x,y = read_csv_truth('%s/kmeanspp.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[2], marker=markers[2], markersize=7,
                label="%s" % legend_list[2] if mi==0 and di==0 else "")
            
            # plot lsa + k-means
            x,y = read_csv_truth('%s/lsa_kmeans.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[4], marker=markers[4], markersize=7,
                label="%s" % legend_list[4] if mi==0 and di==0 else "")

            # plot lsa + dbscan
            x,y = read_csv_truth('%s/lsa_dbscan.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[6], marker=markers[6], markersize=7,
                label="%s" % legend_list[6] if mi==0 and di==0 else "")
            
            # plot se + k-means
            x,y = read_csv_truth('%s/se_kmeans.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[5], marker=markers[5], markersize=7,
                label="%s" % legend_list[5] if mi==0 and di==0 else "")
            
            # plot se + dbscan
            x,y = read_csv_truth('%s/se_dbscan.csv' % dname, 'k', mname)
            ax.plot(x, y, color=colors[7], marker=markers[7], markersize=7,
                label="%s" % legend_list[7] if mi==0 and di==0 else "")
            
            yy = yy_matrix[di][mi]
            xx = xx_list[di]
            plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
            plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        
    plt_helper.plot_fig_legend(ncol=4, font_size=14, legend_width=0.95,
        legend_start=0.99, legend_height=0.4)
    plt_helper.plot_and_save(output_folder, file_name)


# ------------------------------------------------------------------------------
def read_csv_full(file_name):
    '''
    read x,y results from csv file

    :params file_name: file name (string)
    :returns: data (dict)
    '''
    data = {}
    with open(file_name, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        data['k']    = []
        data['mse']  = []
        data['mae']  = []
        data['time'] = []
        data['clus'] = []
        data['m1']   = []
        data['h1']   = []
        data['m2']   = []
        data['h2']   = []
        i = 0
        for row in reader:
            i = i + 1
            if i == 1: continue # skip first row
            
            data['k'].append(int(row[0]))
            data['mse'].append(float(row[1]))
            data['mae'].append(float(row[2]))
            data['time'].append(float(row[3]))
            data['clus'].append(int(row[4]))
            data['m1'].append(int(row[5]))
            data['h1'].append(int(row[6]))
            data['m2'].append(int(row[7]))
            data['h2'].append(int(row[8]))
    # print(data)

    return data


# ------------------------------------------------------------------------------    
def m1_h1_curve(file_name, fig_width, fig_height, dname, output_folder):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.9, bottom_space=0.01, hspace=0.32,
        left_space=0.8, right_space=0.01)
    
    data  = read_csv_full('%s/silk_param.csv' % dname)
    ks    = data['k']
    mses  = data['mse']
    times = data['time']
    m1s   = data['m1']
    h1s   = data['h1']
    m2s   = data['m2']
    h2s   = data['h2']
    
    m1_h1_list = [[8,4], [8,6], [16,4], [16,6], [24,4], [24,6]]
    h2 = 4
    
    # first subfigure
    ax = plt.subplot(1, 2, 1)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'Time (Seconds)')
    
    for line_cnt,m1_h1_pair in enumerate(m1_h1_list):
        m1 = m1_h1_pair[0]
        h1 = m1_h1_pair[1]
        x = []
        y = []
        for a,b,c,d,e in zip(ks,times,m1s,h1s,h2s):
            if (c == m1) and (d == h1) and (e == h2):
                x.append(a)
                y.append(b)
        
        ax.plot(x, y, color=colors[line_cnt], marker=markers[line_cnt], markersize=7,
            label="$L_1=%d$, $K_1=%d$" % (m1, h1))
    
    xx = [0, 10001, 2000]
    yy = [0, 1401,  350]
    plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
    plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    # second subfigure
    ax = plt.subplot(1, 2, 2)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'MSE')
    
    for line_cnt,m1_h1_pair in enumerate(m1_h1_list):
        m1 = m1_h1_pair[0]
        h1 = m1_h1_pair[1]
        x = []
        y = []
        for a,b,c,d,e in zip(ks,mses,m1s,h1s,h2s):
            if (c == m1) and (d == h1) and (e == h2):
                x.append(a)
                y.append(b)
        
        ax.plot(x, y, color=colors[line_cnt], marker=markers[line_cnt], markersize=7)
        
    xx = [0, 10001, 2000]
    yy = [0.124, 0.2001, 0.019]
    plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
    plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    plt_helper.plot_fig_legend(ncol=3, font_size=15, legend_width=0.8,
        legend_start=0.99, legend_height=0.2)
    plt_helper.plot_and_save(output_folder, file_name)


# ------------------------------------------------------------------------------    
def m2_h2_curve(file_name, fig_width, fig_height, dname, output_folder):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.9, bottom_space=0.01, hspace=0.32,
        left_space=0.8, right_space=0.01)
    
    data  = read_csv_full('%s/silk_param.csv' % dname)
    ks    = data['k']
    mses  = data['mse']
    times = data['time']
    m1s   = data['m1']
    h1s   = data['h1']
    m2s   = data['m2']
    h2s   = data['h2']
    
    m2_h2_list = [[6,4], [6,6], [8,4], [8,6], [10,4], [10,6]]
    h1 = 4
    
    # first subfigure
    ax = plt.subplot(1, 2, 1)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'Time (Seconds)')
    
    for line_cnt,m2_h2_pair in enumerate(m2_h2_list):
        m2 = m2_h2_pair[0]
        h2 = m2_h2_pair[1]
        x = []
        y = []
        for a,b,c,d,e in zip(ks,times,h1s,m2s,h2s):
            if (c == h1) and (d == m2) and (e == h2):
                x.append(a)
                y.append(b)
        
        ax.plot(x, y, color=colors[line_cnt], marker=markers[line_cnt], markersize=7,
            label="$L_2=%d$, $K_2=%d$" % (m2, h2))
    
    xx = [0, 10001, 2000]
    yy = [0, 1401,  350]
    plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
    plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    # second subfigure
    ax = plt.subplot(1, 2, 2)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'MSE')
    
    for line_cnt,m2_h2_pair in enumerate(m2_h2_list):
        m2 = m2_h2_pair[0]
        h2 = m2_h2_pair[1]
        x = []
        y = []
        for a,b,c,d,e in zip(ks,mses,h1s,m2s,h2s):
            if (c == h1) and (d == m2) and (e == h2):
                x.append(a)
                y.append(b)
        
        ax.plot(x, y, color=colors[line_cnt], marker=markers[line_cnt], markersize=7)
        
    xx = [0, 10001, 2000]
    yy = [0.124, 0.2121, 0.022]
    plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
    plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    plt_helper.plot_fig_legend(ncol=3, font_size=15, legend_width=0.8,
        legend_start=0.99, legend_height=0.2)
    plt_helper.plot_and_save(output_folder, file_name)


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    output_folder = "../figures/"
    
    # --------------------------------------------------------------------------
    #  plot MSE and MAE vs. global alpha and local alpha
    # --------------------------------------------------------------------------
    fig_width   = 18.0
    fig_height  = 4.3
    file_name   = "choice_alpha"
    dname_list  = ["URL", "Criteo10M", "Avazu", "KDD2012"]
    xx          = [0.1, 0.901, 0.2]
    yy          = [0.1, 0.901, 0.2]
    zz_list     = [
        [0.02, 0.07, 0.01], 
        [0.25, 0.55, 0.06],
        [0.15, 0.41, 0.06], 
        [0.35, 0.45, 0.02]]
    choice_alpha(file_name, dname_list, output_folder, xx, yy, zz_list, 
        fig_width, fig_height)
    
    # --------------------------------------------------------------------------
    #  plot seeding performance
    # --------------------------------------------------------------------------
    fig_width   = 18.0
    xx          = [0, 10001, 2000]
    dname_list  = ["URL", "Criteo10M", "Avazu", "KDD2012"]
    legend_list = ["SILK", "$k$-FreqItems||", "$k$-FreqItems++", "$k$-FreqItems"]
        
    fig_height  = 5.6
    mname_list  = ["wtime", "mse"]
    ylabel_list = ["Time (Seconds)", "MSE"]
    yy_matrix   = [
        [[0, 1601, 400], [0.012, 0.0401, 0.007]], 
        [[0, 2401, 600], [0.237, 0.3411, 0.026]],
        [[0, 4401, 1100], [0.127, 0.2591, 0.033]],
        [[0, 12001, 3000], [0.246, 0.4181, 0.043]]]
    seeding("seeding_no_mae", dname_list, legend_list, output_folder, mname_list, 
        ylabel_list, xx, yy_matrix, fig_width, fig_height)
        
    # fig_height  = 8.0
    # mname_list  = ["wtime", "mse", "mae"]
    # ylabel_list = ["Time (Seconds)", "MSE", "MAE"]
    # yy_matrix   = [
    #     [[0, 1601, 400], [0.012, 0.0401, 0.007], [0.088, 0.1721, 0.021]], 
    #     [[0, 2401, 600], [0.237, 0.3411, 0.026], [0.470, 0.5701, 0.025]],
    #     [[0, 4401, 1100], [0.127, 0.2591, 0.033], [0.333, 0.4931, 0.040]],
    #     [[0, 12001, 3000], [0.246, 0.4181, 0.043], [0.455, 0.6351, 0.045]]]
    # seeding("seeding", dname_list, legend_list, output_folder, mname_list, 
    #     ylabel_list, xx, yy_matrix, fig_width, fig_height)
    
    # --------------------------------------------------------------------------
    #  plot clustering performance
    # --------------------------------------------------------------------------
    fig_width   = 18.0
    xx          = [0, 10001, 2000]
    dname_list  = ["URL", "Criteo10M", "Avazu", "KDD2012"]
    legend_list = ["SILK", "$k$-FreqItems||", "$k$-FreqItems++", "$k$-FreqItems", 
        "SVD + $k$-Means", "SE + $k$-Means"]
    
    fig_height  = 5.6
    mname_list  = ["wtime", "mse"]
    ylabel_list = ["Time (Seconds)", "MSE"]
    yy_matrix   = [
        [[0, 3001,  750],  [0.012, 0.0321, 0.005]], 
        [[0, 5801,  1450], [0.230, 0.3221, 0.023]],
        [[0, 15001, 3750], [0.120, 0.2281, 0.027]],
        [[0, 22001, 5500], [0.245, 0.4131, 0.042]]]
    clustering("clustering_no_mae", dname_list, legend_list, output_folder, 
        mname_list, ylabel_list, xx, yy_matrix, fig_width, fig_height)
        
    # fig_height  = 8.0
    # mname_list  = ["wtime", "mse", "mae"]
    # ylabel_list = ["Time (Seconds)", "MSE", "MAE"]
    # yy_matrix   = [
    #     [[0, 3001,  750],  [0.012, 0.0321, 0.005], [0.088, 0.1521, 0.016]], 
    #     [[0, 5801,  1450], [0.230, 0.3221, 0.023], [0.468, 0.5521, 0.021]],
    #     [[0, 15001, 3750], [0.120, 0.2281, 0.027], [0.328, 0.4561, 0.032]],
    #     [[0, 22001, 5500], [0.245, 0.4131, 0.042], [0.455, 0.6271, 0.043]]]
    # clustering("clustering", dname_list, legend_list, output_folder, mname_list, 
    #     ylabel_list, xx, yy_matrix, fig_width, fig_height)
    
    # --------------------------------------------------------------------------
    #  plot convergence
    # --------------------------------------------------------------------------
    fig_width   = 18.0
    fig_height  = 2.9
    xx          = [0, 11, 2]
    dname_list  = ["URL", "Criteo10M", "Avazu", "KDD2012"]
    file_name   = "convergence"
    mname_list  = ["mse"]
    ylabel_list = ["MSE"]
    
    k = 10000
    yy_matrix = [
        [[0.013, 0.0211, 0.002]], 
        [[0.234, 0.2741, 0.010]],
        [[0.123, 0.1671, 0.011]],
        [[0.253, 0.3051, 0.013]]]
    convergence(file_name, dname_list, k, output_folder, mname_list, 
        ylabel_list, xx, yy_matrix, fig_width, fig_height)
    
    # --------------------------------------------------------------------------
    #  plot multi-gpu performance
    # --------------------------------------------------------------------------
    fig_width   = 9.0
    fig_height  = 5.6 # 9.0
    xx          = [0, 10001, 2000]
    dname_list  = ["Criteo10M", "Avazu"]
    iter_list   = [1, 2, 4, 8]
    mname_list  = ["wtime", "mse"]
    ylabel_list = ["Time (Seconds)", "MSE"]
    
    legend_name = "SILK"
    file_name   = "silk"
    yy_matrix   = [
        [[0, 2001, 500], [0.238, 0.3061, 0.017]],
        [[0, 3601, 900], [0.128, 0.2161, 0.022]]]
    multi_gpus(file_name, legend_name, dname_list, iter_list, output_folder, 
        mname_list, ylabel_list, xx, yy_matrix, fig_width, fig_height)
    
    legend_name = "$k$-FreqItems||"
    file_name   = "kmeansll"
    yy_matrix   = [
        [[0, 6001, 1500], [0.260, 0.3281, 0.017]],
        [[0, 6801, 1700], [0.148, 0.2401, 0.023]]]
    multi_gpus(file_name, legend_name, dname_list, iter_list, output_folder, 
        mname_list, ylabel_list, xx, yy_matrix, fig_width, fig_height)
    
    # --------------------------------------------------------------------------
    #  plot the results of criteo1b
    # --------------------------------------------------------------------------
    fig_width   = 9.0
    fig_height  = 5.6 # 9.0
    xx          = [0, 10001, 2000]
    dname       = "Criteo1B"
    iter_list   = [1, 8]
    title_list  = ["Seeding Performance", "Clustering Performance"]
    mname_list  = ["wtime", "mse"]
    ylabel_list = ["Time (Seconds)", "MSE"]
    yy_matrix   = [
        [[0, 200001, 50000], [0.285, 0.3851, 0.025]],
        [[0, 220001, 55000], [0.272, 0.3481, 0.019]]]
    # mname_list  = ["wtime", "mse", "mae"]
    # ylabel_list = ["Time (Seconds)", "MSE", "MAE"]
    # yy_matrix   = [
    #     [[0, 200001, 40000], [0.285, 0.3851, 0.025], [0.522, 0.6101, 0.022]],
    #     [[0, 300001, 60000], [0.272, 0.3481, 0.019], [0.515, 0.5831, 0.017]]]
    billion(dname, iter_list, title_list, output_folder, mname_list, 
        ylabel_list, xx, yy_matrix, fig_width, fig_height)
    
    # --------------------------------------------------------------------------
    #  plot results with ground truth labels
    # --------------------------------------------------------------------------
    dname_list  = ["News20", "RCV1"]
    mname_list  = ["ari", "purity", "mse"]
    ylabel_list = ["ARI", "Purity", "MSE"]
    xx_list     = [[0, 201, 40], [0, 501, 100]]
    
    fig_width   = 9.0
    fig_height  = 8.3 # 10.7
    legend_list = ["SILK", "$k$-FreqItems||", "$k$-FreqItems++", "$k$-FreqItems",
         "SVD + $k$-Means", "SE + $k$-Means", "SVD + DBSCAN", "SE + DBSCAN"]
    yy_matrix   = [
        [[0.000, 0.1361, 0.034], [0.040, 0.5521, 0.128], [0.752, 0.9721, 0.055]],
        [[0.010, 0.2341, 0.056], [0.222, 0.8301, 0.152], [0.494, 0.9101, 0.104]],
    ]
    truth_full("truth_full_20_200", dname_list, legend_list, output_folder, 
        mname_list, ylabel_list, xx_list, yy_matrix, fig_width, fig_height)
    
    # --------------------------------------------------------------------------
    #  plot parameters on Avazu
    # --------------------------------------------------------------------------
    fig_width  = 9.0
    fig_height = 2.8
    dname      = "Avazu"
    
    m1_h1_curve("param_m1_h1", fig_width, fig_height, dname, output_folder)
    # m2_h2_curve("param_m2_h2", fig_width, fig_height, dname, output_folder)
    
