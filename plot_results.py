import matplotlib.pyplot as plt
import numpy as np


def plot_framework_evolution(framework):
    plt.plot(framework.get_max_trace_accuracy(), label="Maximum")
    plt.plot(framework.get_min_trace_accuracy(), label="Minimum")
    plt.plot(framework.get_avg_trace_accuracy(), label="Average")
    plt.legend()
    plt.xlabel("Framework Iterations")
    plt.ylabel("Single Trace Accuracy")
    plt.show()


def plot_input_gradients_sum(framework, iteration):
    if framework.get_input_gradients_sum() is not None:
        plt.plot(framework.get_input_gradients_sum()[iteration - 1], label="Input Gradients")
        plt.ylabel("Input Gradient")
        plt.xlabel("Sample")
        plt.show()


def plot_input_gradients_epochs(framework, iteration):
    if framework.get_input_gradients_epochs() is not None:
        gradients = framework.get_input_gradients_epochs()[iteration - 1]
        plt.imshow(np.abs(gradients), cmap='jet', interpolation='nearest', aspect='auto')
        plt.ylabel("Epoch")
        plt.xlabel("Sample")
        plt.show()
