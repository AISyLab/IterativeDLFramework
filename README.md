# Iterative Deep Learning-based Framework
This repository provides the source code to reproduce results from paper "Keep it unsupervised: Horizontal Attacks meet Deep Learning" that is accepted on TCHES2021, issue 1. 
Authors: Guilherme Perin, Lukasz Chmielewski, Lejla Batina, Stjepan Picek

## Datasets ##

### Download ###

The datasets to be used with this framework must be downloaded from the following link: https://www.dropbox.com/s/e2mlegb71qp4em3/ecc_datasets.zip?dl=0
The file ecc_datasets.zip contain two ECC datasets. Each dataset contain 76500 traces, each one of them representing a sub-trace in a scalar multiplication. In total, each dataset contains 300 255-bit scalar multiplications. Each scalar multiplication contain a randomized 255-bit scalar.

### Opening datasets ###

To load datasets into numpy arrays, you can use the following example code:

```python
import h5py
import numpy as np

dataset_directory = "C:/my_dir/" # change this path to the location of your downloaded datasets 
dataset_name = "cswap_pointer.h5" # or "cswap_arith.h5"
hf = h5py.File("{}{}".format(dataset_directory, dataset_name), 'r')
train_samples = np.array(hf.get('profiling_traces'))
train_data = np.array(hf.get('profiling_data'))
attack_samples = np.array(hf.get('attacking_traces'))
attack_data = np.array(hf.get('attacking_data'))
```

#### Trace Samples ####

For cswap_pointer.h5 dataset, the variable train_samples is an array of shape (63750, 1000). The total amount of 63750 sub-traces represent 250 255-bit calar multiplication (250 x 255= 63750 sub-traces). The variable attack_samples is an array of shape (12750, 1000). The total amount of 12750 sub-traces represent 50 255-bit calar multiplication (50 x 255= 12750 sub-traces). Each sub-traces contains 1000 samples.

For cswap_arith.h5 dataset, the variable train_samples is an array of shape (63750, 8000). The total amount of 63750 sub-traces represent 250 255-bit calar multiplication (250 x 255= 63750 sub-traces). The variable attack_samples is an array of shape (12750, 8000). The total amount of 12750 sub-traces represent 50 255-bit calar multiplication (50 x 255= 12750 sub-traces). Each sub-traces contains 8000 samples.

#### Trace Data ####

For cswap_pointer.h5 and cswap_arith.h5 datasets, the variable train_data is an array of shape (63750, 2). The total amount of 63750 sub-traces represent 250 255-bit calar multiplication (250 x 255= 63750 sub-traces). The variable attack_data is an array of shape (12750, 2). The total amount of 12750 sub-traces represent 50 255-bit calar multiplication (50 x 255= 12750 sub-traces). Each sub-traces contains 2-byte data. The first second represents the labels (0 or 1) obtained after applying clustering-based horizontal attack. The second byte represents the true label (0 or 1) associated to each sub-trace.

## Code ##

The file __run_framework.py__ contain the main structure to run the iterative framework. This is an example of how the framework is configured and run:

```python
from commons.neural_networks import NeuralNetwork
from commons.framework import Framework

# settings
framework = Framework()
framework.set_directory("D:/traces/ecc_datasets/")
framework.set_dataset("cswap_pointer")
framework.set_znorm()
framework.set_mini_batch(100)
framework.set_epochs(10)
framework.set_neural_network(NeuralNetwork().cnn_cswap_pointer)
framework.run_iterative(
    n_iterations=3,
    data_augmentation=[200],
    visualization=True
)
```

### Plotting framework evolution results ###

To plot the framework results in terms of minimum, maximum and average single (scalar) trace accuracy, you must call the following function:
```python
from plot_results import plot_framework_evolution
plot_framework_evolution(framework)
```

Note that the variable __framework__ is defined in the framework configuration. This function __plot_framework_evolution__ can only be called after the framework is finished. 

