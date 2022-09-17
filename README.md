# $k$-FreqItems: A New Sparse Data Clustering Method Based on Frequent Items

Welcome to the **k-FreqItems** GitHub!

In this repo, we share the implementations and experiments of our work [A New Sparse Data Clustering Method Based on Frequent Items](https://drive.google.com/file/d/1WyqXT0XOzy62MEK7hF_5mBkgNxuzowKT/view?usp=sharing) in [SIGMOD 2023](https://2023.sigmod.org/index.shtml). We implement k-FreqItems and SILK with CUDA on a distributed platform (one master with 4 GPUs and one slave with 4 GPUs) for clustering massive sparse data sets on Jaccard distance. We also adapt two state-of-the-art methods [k-Means++](https://dl.acm.org/doi/abs/10.5555/1283383.1283494) and [k-Means$\parallel$](https://dl.acm.org/doi/abs/10.14778/2180912.2180915) for Jaccard distance as baselines.

## Data Sets

We choose two small sparse data sets with ground truth laebls (i.e., [News20](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#news20) and [RCV1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#rcv1.multiclass)) and five large-scale real-world sparse data sets (i.e., [URL](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#url), [Avazu](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#avazu), [KDD2012](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2012), [Criteo10M and Criteo1B](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#criteo_tb)) for performance evaluation.Users can download the data sets from their links. The statistics of data sets are summarized as follows.

| Data Sets | # Data            | # Dim             | # Non-Zero Dim | Data Size | Global $\alpha$ | Local $\alpha$ |
| --------- | ----------------- | ----------------- | -------------- | --------- | --------------- | -------------- |
| News20    | $2.0 \times 10^4$ | $6.2 \times 10^4$ | 80             | 6.3 MB    | 0.2             | -              |
| RCV1      | $5.3 \times 10^5$ | $4.7 \times 10^4$ | 65             | 136 MB    | 0.2             | -              |
| URL       | $2.3 \times 10^6$ | $2.3 \times 10^6$ | 116            | 1.1 GB    | 0.4             | 0.2            |
| Criteo10M | $1.0 \times 10^7$ | $1.0 \times 10^6$ | 39             | 1.6 GB    | 0.2             | 0.1            |
| Avazu     | $4.0 \times 10^7$ | $1.0 \times 10^6$ | 15             | 2.6 GB    | 0.3             | 0.2            |
| KDD2012   | $1.5 \times 10^8$ | $5.4 \times 10^7$ | 11             | 7.3 GB    | 0.5             | 0.2            |
| Criteo1B  | $1.0 \times 10^9$ | $1.0 \times 10^6$ | 39             | 153 GB    | 0.2             | 0.1            |

## Sparse Data Format

The input sparse data sets are stored in binary format, where each file consists of two fields: `pos` and `data`, as shown below:

| field | field type       | description                             |
| ----- | ---------------- | --------------------------------------- |
| pos   | `uint64_t`*(n+1) | start position of `data` for each point |
| data  | `int`*pos[n]     | non-zero dimensions IDs of all points   |

`pos` is an `uint64_t` array of (n+1) length, which stores the start position of the `data` array for each sparse data point. `data` is an `int` array of pos[n] length, which store the non-zero dimensions IDs of all sparse data points. Here we assume that all non-zero dimension IDs can be represented by an `int` type integer. If the dimensionality exceeds the range of `int`, one can store the non-zero dimension IDs by `uint64_t` type with minor modification.With `pos` and `data`, one can efficient retrieve a specific data point with its data ID.

For example, suppose there is a sparse data set with four points: `x_0={1,3,5,8}`, `x_1={1,3}`, `x_2={1,6,8}`, and `x_3={1,8,10}`. Then, the `pos` array is `[0,4,6,9,12]`, e.g., pos[0]=0, pos[4]=12. And the `data` array is `[1,3,5,8,1,3,1,6,8,1,8,10]`. If you want to retrieve `x_1`, you can first get its start position of `data` and its length from `pos` by its data ID `1`, i.e., start position is `pos[1]=4`, and its length is `pos[1+1]-pos[1]=6-4=2`. Then you can retrieve `x_1` from `data` by the start position `4` and its length `2`, i.e., `x_1={1,3}`.

We also provide the transformation code in the folder `transformation/` to convert sparse data sets into the binary format we use here.

## Compilation

The source codes require ```nvcc``` with ```c++11``` support. We have provided `Makefile` for compilation. However, for different machines, users may need to specify your local path to MPI libary and CUDA SM and Gencode variations. Suppose the `Makefile` is in accordance with the machines, users can use the following commands to compile the source codes:

```bash
cd methods/
make clean
make -j
```

Representative NVIDIA GPU cards for each architecture name and CUDA version can be found in <http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>.

## Usages

We provide bash (and python) scripts to reproduce all experiments reported in our SIGMOD 2023 paper. Suppose you have cloned this repo and you are in the folder `k-FreqItems/`.

### Step 1: Prepare Data Sets

Please first download the datasets from the links we provided. Then, you can use `transformation/sparse.cc` to convert them into the binary format we support. If you want to try multiple GPU and/or nodes, you could further run `tranformation/partitin.cc` to split the data into 1,2,4,8 slices. After that, you copy the binary files to a specific folder like `data/`. For example, when you get URL_0.bin, you can move it to the path `data/URL/URL_0.bin`.

### Step 2: Reproduce Experiments

After preparing the data sets, we provide bash scripts to run k-FreqItems, SILK, and two baselines k-FreqItems++ and k-FreqItems$\parallel$. Users can reproduce the experiments by simply running the following command:

```bash
cd methods/
bash run.sh
```

By default, we provide the scripts for running methods with 4 GPUs once users specify the data path. With some minor modifications, users can run methods with different number of GPUs based on their machines.

### Step 3: Visualizations

We provide python scripts (i.e., `plot.py` and `plot_util.py` in `scripts/`) to reproduce all the figures that are appeared in our SIGMOD 2023 paper. These scripts require python 3.7 (or higher versions) with numpy, scipy, and matplotlib installed. If not, you might need to use anaconda to create a new virtual environment and use pip to install those packages.

With the experimental results from step 2, users can reproduce all the figures with the following commands.

```python
cd scripts/
python plot.py
```

### Step 4: Experiments for Center Representation

Finally, we provide the source codes and results for the validation of FreqItem representation (Section 3.2 in [our SIGMOD 2023 paper](https://drive.google.com/file/d/1WyqXT0XOzy62MEK7hF_5mBkgNxuzowKT/view?usp=sharing)). Please refer to the folder `representation/`.

## Reference

Thank you so much for being so patient to read the user manual. We will appreciate using the following BibTeX to cite this work when you use k-FreqItems (or SILK) in your paper.

```tex
@inproceedings{huang2023a,
  title={Point-to-Hyperplane Nearest Neighbor Search Beyond the Unit Hypersphere},
  author={Huang, Qiang and Luo, Pingyi and Tung, Anthony KH},
  booktitle={Proceedings of the 2023 International Conference on Management of Data (SIGMOD)},
  year={2023}
}
```

It is welcome to contact me (huangq@comp.nus.edu.sg) if you meet any issue. Thank you.
