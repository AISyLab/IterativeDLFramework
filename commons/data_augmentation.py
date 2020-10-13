import random
import numpy as np


class DataAugmentation:

    def __init__(self):
        self.std = None
        self.mean = None

    def initialize_parameters(self, x, param):
        samples = np.zeros((param["number_of_samples"], len(x)))
        self.std = np.zeros(param["number_of_samples"])
        for j in range(param["number_of_samples"]):
            for i in range(len(x)):
                samples[j][i] = x[i][j]
            self.std[j] = np.std(samples[j])

    # ---------------------------------------------------------------------------------------------------------------------#
    #  Functions for Data Augmentation
    # ---------------------------------------------------------------------------------------------------------------------#
    def data_augmentation_shifts(self, data_set_samples, data_set_labels, param):
        ns = param["number_of_samples"]

        while True:

            x_train_shifted = np.zeros((param["mini-batch"], ns))
            rnd = random.randint(0, len(data_set_samples) - param["mini-batch"])
            x_mini_batch = data_set_samples[rnd:rnd + param["mini-batch"]]

            for trace_index in range(param["mini-batch"]):
                x_train_shifted[trace_index] = x_mini_batch[trace_index]
                shift = random.randint(-5, 5)
                if shift > 0:
                    x_train_shifted[trace_index][0:ns - shift] = x_mini_batch[trace_index][shift:ns]
                    x_train_shifted[trace_index][ns - shift:ns] = x_mini_batch[trace_index][0:shift]
                else:
                    x_train_shifted[trace_index][0:abs(shift)] = x_mini_batch[trace_index][ns - abs(shift):ns]
                    x_train_shifted[trace_index][abs(shift):ns] = x_mini_batch[trace_index][0:ns - abs(shift)]

            x_train_shifted_reshaped = x_train_shifted.reshape((x_train_shifted.shape[0], x_train_shifted.shape[1], 1))
            yield x_train_shifted_reshaped, data_set_labels[rnd:rnd + param["mini-batch"]]

    def data_augmentation_noise(self, data_set_samples, data_set_labels, param):
        ns = param["number_of_samples"]

        while True:

            x_train_augmented = np.zeros((param["mini-batch"], ns))
            rnd = random.randint(0, len(data_set_samples) - param["mini-batch"])
            x_mini_batch = data_set_samples[rnd:rnd + param["mini-batch"]]

            noise = np.random.normal(0, 1, param["number_of_samples"])

            for trace_index in range(param["mini-batch"]):
                x_train_augmented[trace_index] = x_mini_batch[trace_index] + noise

            x_train_augmented_reshaped = x_train_augmented.reshape((x_train_augmented.shape[0], x_train_augmented.shape[1], 1))
            yield x_train_augmented_reshaped, data_set_labels[rnd:rnd + param["mini-batch"]]
