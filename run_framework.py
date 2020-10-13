import os
import tensorflow as tf
from plot_results import plot_framework_evolution, plot_input_gradients_epochs, plot_input_gradients_sum

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
    visualization=True,
    equal_size_traces=True
)

# plot
plot_framework_evolution(framework)
plot_input_gradients_sum(framework, 3)
plot_input_gradients_epochs(framework, 3)


