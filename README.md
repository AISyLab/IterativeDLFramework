# IterativeDLFramework
This repository provides the source code to reproduce results from paper "Keep it unsupervised: Horizontal Attacks meet Deep Learning" that is accepted on TCHES2021, issue 1. 

# Datasets

The datasets to be used with this framework must be downloaded from the following link: https://www.dropbox.com/s/e2mlegb71qp4em3/ecc_datasets.zip?dl=0
The file ecc_datasets.zip contain two ECC datasets. Each dataset contain 76500 traces, each one of them representing a sub-trace in a scalar multiplication. In total, each dataset contains 300 255-bit scalar multiplications. Each scalar multiplication contain a randomized 255-bit scalar.

To load datasets into numpy arrays, you can use the following example code:

```python
import h5py
dataset_directory = "C:/my_dir/" # change this path to the location of your downloaded datasets 
dataset_name = "cswap_pointer.h5" # or "cswap_arith.h5"
hf = h5py.File("{}{}".format(dataset_directory, dataset_name), 'r')
train_samples = np.array(hf.get('profiling_traces'))
train_data = np.array(hf.get('profiling_data'))
```
