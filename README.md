# Iterative Deep Learning-based Framework
This repository provides the source code to reproduce results from paper "__Keep it unsupervised: Horizontal Attacks meet Deep Learning__" that is accepted on TCHES2021, issue 1. 

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
    n_iterations=50,
    data_augmentation=[200],
    visualization=True
)
```

As we can see in the above code, the user can set the directory to where the datasets (in our case, __cswap_pointer.h5__ and __cswap_arith.h5__) are located. By setting the dataset with method __set_dataset__, parameters from __commons/datasets.py__ are automatically set to the framework. For example, when we set __cswap_pointer__ as the dataset, the following parameters are defined:

```python
parameters_cswap_pointer = {
    "name": "cswap_pointer",
    "data_length": 2,
    "first_sample": 0,
    "number_of_samples": 1000,
    "n_set1": 31875,
    "n_set2": 31875,
    "n_attack": 12750,
    "classes": 2,
    "epochs": 25,
    "mini-batch": 64
}
```

To add a new dataset, please check the section __Adding new datasets__.

### Plotting framework evolution results ###

To plot the framework results in terms of minimum, maximum and average single (scalar) trace accuracy, you must call the following function:
```python
from plot_results import plot_framework_evolution
plot_framework_evolution(framework)
```

Note that the variable __framework__ is defined in the framework configuration. This function __plot_framework_evolution__ can only be called after the framework is finished. 

### Plotting sum of input gradients ###

To plot the sum of input gradients (for all the epochs of one framework iteration __in phase 3__), the user must call the following method:
```python
from plot_results import plot_input_gradients_sum
plot_input_gradients_sum(framework, 10)
```

In the above example, it will generate a plot for the sum of input gradients from all epochs during the neural network training in iteration 10.

### Plotting input gradients for all epochs ###

To plot input gradients for all the epochs of one framework iteration (__in phase 3__), the user must call the following method:
```python
from plot_results import plot_input_gradients_epochs
plot_input_gradients_epochs(framework, 10)
```

In the above example, it will generate a 2D plot for the input gradients for all epochs during the neural network training in iteration 10.

## Convolutional Neural Networks ##

The file __commons/neural_networks.py__ contain all CNN models used in the paper. They are:

CNN model name  | Details
------------- | -------------
cnn_supervised  | CNN used to produce supervised classification results
cnn_cswap_pointer  | CNN without regularization used to attack __cswap_pointer__ dataset
cnn_cswap_pointer_dropout  | CNN with dropout used to attack __cswap_pointer__ dataset
cnn_cswap_pointer_random  | CNN with hyperparameters defined at random without regularization used to attack __cswap_pointer__ dataset
cnn_cswap_pointer_dropout_random  | CNN with hyperparameters defined at random with dropout used to attack __cswap_pointer__ dataset
cnn_cswap_arith  | CNN without regularization used to attack __cswap_arith__ dataset
cnn_cswap_arith_dropout  | CNN with dropout used to attack __cswap_arith__ dataset
cnn_cswap_arith_random  | CNN with hyperparameters defined at random without regularization used to attack __cswap_arith__ dataset
cnn_cswap_arith_dropout_random  | CNN with hyperparameters defined at random with dropout used to attack __cswap_arith__ dataset

Furthermore, we also define CNNs for the case when attacked interval is narrowed down by using gradient visualization:

CNN model name  | Details
------------- | -------------
cnn_cswap_pointer_gv  | CNN without regularization used to attack __cswap_pointer__ dataset
cnn_cswap_pointer_dropout_gv  | CNN with dropout used to attack __cswap_pointer__ dataset
cnn_cswap_pointer_random_gv  | CNN with hyperparameters defined at random without regularization used to attack __cswap_pointer__ dataset
cnn_cswap_pointer_dropout_random_gv  | CNN with hyperparameters defined at random with dropout used to attack __cswap_pointer__ dataset
cnn_cswap_arith_gv  | CNN without regularization used to attack __cswap_arith__ dataset
cnn_cswap_arith_dropout_gv  | CNN with dropout used to attack __cswap_arith__ dataset
cnn_cswap_arith_random_gv  | CNN with hyperparameters defined at random without regularization used to attack __cswap_arith__ dataset
cnn_cswap_arith_dropout_random_gv  | CNN with hyperparameters defined at random with dropout used to attack __cswap_arith__ dataset

## Adding new datasets ##

To add a new dataset to the framework, the user must only edit the file __commons/datasets.py__. First, add a new dataset to the 

```python
self.dataset_list = {
    "cswap_arith": parameters_cswap_arith,
    "cswap_pointer": parameters_cswap_pointer,
    "my_dataset": parameters_my_dataset,
}
```

Second, the user must define the dictionary for the new dataset:

```python
parameters_my_dataset = {
    "name": "my_dataset",
    "data_length": 2,
    "first_sample": 0,
    "number_of_samples": 2000,
    "n_set1": 25500,
    "n_set2": 25500,
    "n_attack": 510,
    "classes": 2,
    "epochs": 25,
    "mini-batch": 64
}
```

Now, __my_dataset__ can be called from __set_dataset()__ method.

## Additional information ##

For the two considered datasets, __cswap_pointer__ and __cswap_arith__, all scalar multiplications contain exactly 255 bits, resulting in 255 sub-traces per scalar multiplication. In case your dataset contains scalar multiplication (or modular exponentiation) traces with different bit lengths, you need to set the parameter __equal_sizes_traces=False__ before running the framework (this parameter is set to __True__ as a default value):

```python
framework.run_iterative(
    n_iterations=50,
    data_augmentation=[200],
    visualization=True,
    equal_size_traces=False
)
```

This way, the information displayed during training will provide the accuracy for the while attack set, instead per scalar multiplication (or modular exponentiation) trace. After the framework finished the execution, the following method returns the final label for each sub-trace in the full attack set, for all iterations:

```python
labels_dl = framework.get_labels_dl()
```

Then, the user can, e.g., select __labels_dl[10]__ that are all new labels returned by the iterative framework after phase 3 of iteration 10, in all the epochs. As an example __labels_dl[10][5]__ returns the labels for the full attack set after phase 3 of iteration 10 and epoch 5. After that, the user can split the vector __labels_dl[10][5]__ into scalar multiplication (or modular exponentiation) traces with different lenghts. 

Additionaly, the user can also retrive the accuracy for set1 and for set2 for all framework iterations:
```python
accuracy_set1 = framework.get_accuracy_set1()
accuracy_set2 = framework.get_accuracy_set2()
```

The vectors __accuracy_set1__ and __accuracy_set2__ contain the accuracy for the full set1 and set2 after the processing of framework iteration phases 2 and 3, respectively. 
