import os
import sys
import inspect

import math
import json
import csv
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt

from plot_util import PlotHelper

markers = ['s','o','^','x','D','v','>','o','v','^','s','D','x','>']
colors  = ['crimson', 'blueviolet', 'forestgreen', 'blue', 
           'limegreen', 'royalblue', 'orange',
           'red','cyan', 'lime', 'purple', 'olive', 
           'deeppink', 'royalblue', 'forestgreen', 'violet', 'glod']


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
        data['center'] = []
        data['mse']    = []
        data['mae']    = []
        data['time']   = []
        i = 0
        for row in reader:
            i = i + 1
            if i==105 or i==108 or i==111 or i==114:
                data['center'].append(row[0])
                data['mse'].append(float(row[1]))
                data['mae'].append(float(row[2]))
                data['time'].append(float(row[3]))
            else:
                continue
    # print(data)
    
    return data[x_name], data[y_name]


# ------------------------------------------------------------------------------
def load(m, input_folder, mname, labels):

    freqitem = []
    medoid   = []
    mode1    = []
    mode2    = []
    for label in labels:
        fname = "%s%s_%d_1.csv" % (input_folder, label, m)
        print(fname)
        x_list, y_list = read_csv(fname, "center", mname)
        # print(x_list)
        for x,y in zip(x_list, y_list):
            if x == "FreqItem": freqitem.append(y)
            elif x == "Medoid": medoid.append(y)
            elif x == "Mode1":  mode1.append(y)
            elif x == "Mode2":  mode2.append(y)
            else: print("ERROR!\n"); exit

    return freqitem, medoid, mode1, mode2


# ------------------------------------------------------------------------------
def histogram(output_name, m, input_folder, output_folder, mname_list, 
    ylabel_list, labels, fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.66, hspace=0.37)
    plt.rcParams['hatch.linewidth'] = 1.2
    
    for mi,(mname,ylabel) in enumerate(zip(mname_list, ylabel_list)):
        freqitem, medoid, mode1, mode2 = load(m, input_folder, mname, labels)
            
        x = np.arange(len(labels))  # the label locations
        width = 0.16  # the width of the bars
        
        ax = plt.subplot(1, len(mname_list), mi+1)
        ax.bar(x - width*1.5, freqitem, width, label="FreqItem" if mi==0 else "",
            color="#EB0000", edgecolor="black", linewidth=0.5, hatch='///')
        
        ax.bar(x - width*0.5, medoid, width, label="Medoid" if mi==0 else "", 
            color="mediumpurple", edgecolor="black", linewidth=0.5, hatch='..')
        
        ax.bar(x + width*0.5, mode1, width, label="Mode1" if mi==0 else "", 
            color="lightskyblue", edgecolor="black", linewidth=0.5, hatch='xxx')
        
        ax.bar(x + width*1.5, mode2, width, label="Mode2" if mi==0 else "", 
            color="lightgreen", edgecolor="black", linewidth=0.5, hatch='**')
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('%s' % ylabel)
        # ax.set_title('%s by different representations' % ylabel, fontsize=14)
        plt.xticks(x, labels)
        plt.yticks(np.arange(0.0, 1.01, step=0.25))
        # plt.grid(color='grey', linestyle='-.', linewidth=0.5)
        
    plt_helper.plot_fig_legend(ncol=4, legend_width=0.7)
    plt_helper.plot_and_save(output_folder, "%s_%d" % (output_name, m))


# ------------------------------------------------------------------------------
def read_csv_100_steps(file_name):
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
        data['alpha'] = []
        data['mse']   = []
        data['mae']   = []
        data['time']  = []
        i = 0
        for row in reader:
            i = i + 1
            if i>=2 and i<=102:
                data['alpha'].append(float(row[0]))
                data['mse'].append(float(row[1]))
                data['mae'].append(float(row[2]))
                data['time'].append(float(row[3]))
            else:
                continue
    # print(data)
    
    return data['alpha'], data['mse'], data['mae']


# ------------------------------------------------------------------------------
def read_csv_10_steps(file_name):
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
        data['alpha'] = []
        data['mse']   = []
        data['mae']   = []
        data['time']  = []
        i = 0
        for row in reader:
            i = i + 1
            if i>=2 and i<=12:
                data['alpha'].append(float(row[0]))
                data['mse'].append(float(row[1]))
                data['mae'].append(float(row[2]))
                data['time'].append(float(row[3]))
            else:
                continue
    # print(data)
    
    return data['alpha'], data['mse'], data['mae']
    

# ------------------------------------------------------------------------------
def curve(fname, m, input_folder, output_folder, dname_list, xx, yy_list, 
    fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.9, hspace=0.37)
    
    for di,dname, in enumerate(dname_list):
        ax = plt.subplot(2, 2, di+1)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'MSE / MAE')
        ax.set_title('%s' % dname)

        # plot mse and mae
        x,y1,y2 = read_csv_100_steps('%s%s_%d_1.csv' % (input_folder, dname, m))
        ax.plot(x, y1, color="red",  linestyle='solid',  label="MSE" if di==0 else "")
        ax.plot(x, y2, color="blue", linestyle='dashed', label="MAE" if di==0 else "")
    
        # x,y1,y2 = read_csv_10_steps('backup/results/%s_%d.csv' % (dname, m))
        # ax.plot(x, y1, color="red", marker='s', markersize=7, label="MSE" if di==0 else "")
        # ax.plot(x, y2, color="blue", marker='o', markersize=7, label="MAE" if di==0 else "")
        
        yy = yy_list[di]
        plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
        plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
        plt.grid(color='grey', linestyle='--', linewidth=1.0)
        
    plt_helper.plot_fig_legend(ncol=4, legend_width=0.5)
    plt_helper.plot_and_save(output_folder, 'FreqItem_%s_%d' % (fname,m))

# ------------------------------------------------------------------------------
def curve2(fname, m, input_folder, output_folder, dname_list, xx, yy_list, 
    fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.9, hspace=0.37)
    
    for di,dname, in enumerate(dname_list):
        ax = plt.subplot(2, 2, di+1)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'MSE')
        ax.set_title('%s' % dname)

        # plot mse and mae
        x,y1,y2 = read_csv_100_steps('%s%s_%d_1.csv' % (input_folder, dname, m))
        ax.plot(x, y1, color="blue",  linestyle='solid',  label="MSE" if di==0 else "")
        # ax.plot(x, y2, color="blue", linestyle='dashed', label="MAE" if di==0 else "")
            
        yy = yy_list[di]
        plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
        plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
        plt.grid(color='grey', linestyle='--', linewidth=1.0)
        
    # plt_helper.plot_fig_legend(ncol=4, legend_width=0.5)
    plt_helper.plot_and_save(output_folder, 'FreqItem_%s_%d' % (fname,m))


# ------------------------------------------------------------------------------
def stable(m, input_folder, output_folder, dname, xx, yy, fig_width, fig_height):
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.9, hspace=0.37)
    
    id_list = [2,3,4,5]
    for di,id, in enumerate(id_list):
        ax = plt.subplot(2, 2, di+1)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'MSE / MAE')
        ax.set_title('%s' % dname)

        # plot mse and mae
        x,y1,y2 = read_csv_100_steps('%s%s_%d_%d.csv' % (input_folder, dname, m, id))
        ax.plot(x, y1, color="red",  linestyle='solid',  label="MSE" if di==0 else "")
        ax.plot(x, y2, color="blue", linestyle='dashed', label="MAE" if di==0 else "")
        
        plt.yticks(np.arange(yy[0], yy[1], step=yy[2]))
        plt.xticks(np.arange(xx[0], xx[1], step=xx[2]))
        plt.grid(color='grey', linestyle='--', linewidth=1.0)
        
    plt_helper.plot_fig_legend(ncol=4, legend_width=0.5)
    plt_helper.plot_and_save(output_folder, 'Stable_%s_%d' % (dname,m))


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    
    m=1000
    input_folder  = "results/"
    output_folder = "figures/"
    labels = ["URL", "Criteo10M", "Avazu", "KDD2012"]
    xx = [0.0, 1.01, 0.2]
    
    # 1. plot the histogram of mse and mae
    fig_width   = 9.5
    fig_height  = 3.3 # 3.0
    mname_list  = ["mse"]
    ylabel_list = ["MSE"]
    histogram("Representation_no_mae", m, input_folder, output_folder, 
        mname_list, ylabel_list, labels, fig_width, fig_height)
    
    # mname_list  = ["mse", "mae"]
    # ylabel_list = ["MSE", "MAE"]
    # histogram("Representation", m, input_folder, output_folder, mname_list, 
    #     ylabel_list, labels, fig_width, fig_height)
    
    # 2. plot the curve of mse and mae
    labels = ["Criteo10M", "Avazu"]
    fig_width  = 9.0
    fig_height = 5.5 # 5.8
    yy_list = [
        [0.56, 1.01, 0.11], # Criteo10M
        [0.56, 1.01, 0.11]] # Avazu
    curve2("Criteo10M_and_Avazu_no_mae", m, input_folder, output_folder, labels, xx, 
        yy_list, fig_width, fig_height)
    
    # curve("Criteo10M_and_Avazu", m, input_folder, output_folder, labels, xx, 
    #     yy_list, fig_width, fig_height)


    # --------------------------------------------------------------------------
    #  other curves that might be useful for validation
    # --------------------------------------------------------------------------
    # # 3. plot the curve of mse and mae
    # fig_width  = 9.0 # 18.0
    # fig_height = 6.5 # 3.5
    # yy_list = [
    #     [0.0,  1.01, 0.25], # URL
    #     [0.56, 1.01, 0.11], # Criteo10M
    #     [0.56, 1.01, 0.11], # Avazu
    #     [0.6,  1.01, 0.1]]  # KDD2012
    # curve("ALL", m, input_folder, output_folder, labels, xx, yy_list, fig_width, 
    #     fig_height)
    
    # # 4. plot the curve of mse and mae
    # fig_width  = 9.0 # 18.0
    # fig_height = 6.5 # 3.5
    # labels = ["URL", "Criteo10M", "Avazu", "KDD2012"]
    # yy_list = [
    #     [0.0,  1.01, 0.25], # URL
    #     [0.56, 1.01, 0.11], # Criteo10M
    #     [0.56, 1.01, 0.11], # Avazu
    #     [0.6,  1.01, 0.1]]  # KDD2012
    # for label,yy in zip(labels, yy_list):
    #     stable(m, input_folder, output_folder, label, xx, yy, fig_width, fig_height)
    