from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as backend
import tensorflow as tf
import numpy as np
from termcolor import colored


class TestCallback(Callback):
    def __init__(self, x_data, y_data):
        self.current_epoch = 0
        self.x = x_data
        self.y = y_data
        self.accuracy = []
        self.recall = []
        self.loss = []

    def on_epoch_end(self, epoch, logs={}):
        loss, acc, recall = self.model.evaluate(self.x, self.y, verbose=0)

        self.accuracy.append(acc)
        self.recall.append(recall)
        self.loss.append(loss)

    def get_accuracy(self):
        return self.accuracy

    def get_recall(self):
        return self.recall

    def get_loss(self):
        return self.loss


class CalculateAccuracyCurve25519(Callback):
    def __init__(self, x_data, y_true, y_ha, number_of_epochs):
        self.x_data = x_data
        self.labels_true = y_true
        self.labels_ha = y_ha
        self.max_accuracy = 0
        self.min_accuracy = 1
        self.avg_accuracy = 0
        self.max_accuracy_per_epoch = np.zeros(number_of_epochs)
        self.min_accuracy_per_epoch = np.zeros(number_of_epochs)
        self.accuracy_per_epoch = np.zeros(number_of_epochs)
        self.nbits_per_trace = 255
        self.number_of_validation_traces = int(len(self.labels_true) / self.nbits_per_trace)
        self.correct_bits_dl = np.zeros((self.number_of_validation_traces, number_of_epochs))
        self.correct_bits_ha = np.zeros((self.number_of_validation_traces, number_of_epochs))
        self.max_ttest = []
        self.x_samples = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1])
        self.labels_1 = []
        self.epochs = number_of_epochs
        self.labels_dl = np.zeros((number_of_epochs, len(x_data)))

    def on_epoch_end(self, epoch, logs=None):

        output_probabilities = self.model.predict(self.x_data)

        for bit_index in range(len(output_probabilities)):

            if output_probabilities[bit_index][1] > 0.5:
                self.labels_dl[epoch][bit_index] = 1
            else:
                self.labels_dl[epoch][bit_index] = 0

        for trace_index in range(self.number_of_validation_traces):
            trace_probabilities = output_probabilities[
                                  trace_index * self.nbits_per_trace:(trace_index + 1) * self.nbits_per_trace]
            trace_labels_true = self.labels_true[
                                trace_index * self.nbits_per_trace:(trace_index + 1) * self.nbits_per_trace]
            trace_labels_ha = self.labels_ha[
                              trace_index * self.nbits_per_trace:(trace_index + 1) * self.nbits_per_trace]

            labels_1 = 0

            for bit_index in range(len(trace_probabilities)):
                if trace_probabilities[bit_index][1] > 0.5 and trace_labels_true[bit_index] == 1:
                    self.correct_bits_dl[trace_index][epoch] += 1
                    labels_1 += 1
                if trace_probabilities[bit_index][0] > 0.5 and trace_labels_true[bit_index] == 0:
                    self.correct_bits_dl[trace_index][epoch] += 1
                if trace_labels_true[bit_index] == trace_labels_ha[bit_index]:
                    self.correct_bits_ha[trace_index][epoch] += 1

            if self.correct_bits_dl[trace_index][epoch] / self.nbits_per_trace > 0.6:
                print(colored("Correct rate trace {} :{} / ha: {} (bits 1: {}, bits 0: {})".format(trace_index,
                                                                                                   self.correct_bits_dl[trace_index][
                                                                                                       epoch] / self.nbits_per_trace,
                                                                                                   self.correct_bits_ha[trace_index][
                                                                                                       epoch] / self.nbits_per_trace,
                                                                                                   labels_1,
                                                                                                   self.nbits_per_trace - labels_1),
                              'green'))
            else:
                print("Correct rate trace {} :{} / ha: {} (bits 1: {}, bits 0: {})".format(trace_index,
                                                                                           self.correct_bits_dl[trace_index][
                                                                                               epoch] / self.nbits_per_trace,
                                                                                           self.correct_bits_ha[trace_index][
                                                                                               epoch] / self.nbits_per_trace,
                                                                                           labels_1, self.nbits_per_trace - labels_1))

            self.correct_bits_dl[trace_index][epoch] /= self.nbits_per_trace
            self.correct_bits_ha[trace_index][epoch] /= self.nbits_per_trace

            if self.correct_bits_dl[trace_index][epoch] > self.max_accuracy:
                self.max_accuracy = self.correct_bits_dl[trace_index][epoch]

            if self.correct_bits_dl[trace_index][epoch] < self.min_accuracy:
                self.min_accuracy = self.correct_bits_dl[trace_index][epoch]

            self.max_accuracy_per_epoch[epoch] = self.max_accuracy
            self.min_accuracy_per_epoch[epoch] = self.min_accuracy

            if epoch == self.epochs - 1:
                self.avg_accuracy += self.correct_bits_dl[trace_index][epoch]

        self.avg_accuracy /= self.number_of_validation_traces

        print("\nMax Accuracy: " + str(self.max_accuracy))

    def get_labels_dl(self):
        return self.labels_dl

    def get_max_accuracy(self):
        return self.max_accuracy

    def get_min_accuracy(self):
        return self.min_accuracy

    def get_avg_accuracy(self):
        return self.avg_accuracy

    def get_max_accuracy_per_epoch(self):
        return self.max_accuracy_per_epoch

    def get_min_accuracy_per_epoch(self):
        return self.min_accuracy_per_epoch

    def get_accuracy_per_epoch(self):
        return self.accuracy_per_epoch


class CalculateAccuracy(Callback):
    def __init__(self, x_data, y_true, y_ha, number_of_epochs):
        self.x_data = x_data
        self.labels_true = y_true
        self.labels_ha = y_ha
        self.max_accuracy = 0
        self.min_accuracy = 1
        self.avg_accuracy = 0
        self.max_accuracy_per_epoch = np.zeros(number_of_epochs)
        self.min_accuracy_per_epoch = np.zeros(number_of_epochs)
        self.accuracy_per_epoch = np.zeros(number_of_epochs)
        self.correct_bits_dl = np.zeros(number_of_epochs)
        self.correct_bits_ha = np.zeros(number_of_epochs)
        self.x_samples = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1])
        self.labels_1 = []
        self.epochs = number_of_epochs
        self.labels_dl = np.zeros((number_of_epochs, len(x_data)))

    def on_epoch_end(self, epoch, logs=None):

        output_probabilities = self.model.predict(self.x_data)

        labels_1 = 0

        for bit_index in range(len(output_probabilities)):

            if output_probabilities[bit_index][1] > 0.5:
                self.labels_dl[epoch][bit_index] = 1
            else:
                self.labels_dl[epoch][bit_index] = 0

            if output_probabilities[bit_index][1] > 0.5 and self.labels_true[bit_index] == 1:
                self.correct_bits_dl[epoch] += 1
                labels_1 += 1

            if output_probabilities[bit_index][0] > 0.5 and self.labels_true[bit_index] == 0:
                self.correct_bits_dl[epoch] += 1

            if self.labels_true[bit_index] == self.labels_ha[bit_index]:
                self.correct_bits_ha[epoch] += 1

        if self.correct_bits_dl[epoch] / len(output_probabilities) > 0.6:
            print(colored(
                "Correct rate :{} / ha: {} (bits 1: {}, bits 0: {})".format(self.correct_bits_dl[epoch] / len(output_probabilities),
                                                                            self.correct_bits_ha[epoch] / len(output_probabilities),
                                                                            labels_1, len(output_probabilities) - labels_1), 'green'))
        else:
            print("Correct rate :{} / ha: {} (bits 1: {}, bits 0: {})".format(self.correct_bits_dl[epoch] / len(output_probabilities),
                                                                              self.correct_bits_ha[epoch] / len(output_probabilities),
                                                                              labels_1, len(output_probabilities) - labels_1))

        self.correct_bits_dl[epoch] /= len(output_probabilities)
        self.correct_bits_ha[epoch] /= len(output_probabilities)

        if self.correct_bits_dl[epoch] > self.max_accuracy:
            self.max_accuracy = self.correct_bits_dl[epoch]

        if self.correct_bits_dl[epoch] < self.min_accuracy:
            self.min_accuracy = self.correct_bits_dl[epoch]

        self.max_accuracy_per_epoch[epoch] = self.max_accuracy
        self.min_accuracy_per_epoch[epoch] = self.min_accuracy

        if epoch == self.epochs - 1:
            self.avg_accuracy += self.correct_bits_dl[epoch]

        self.avg_accuracy /= len(output_probabilities)

        print("\nMax Accuracy: " + str(self.max_accuracy))

    def get_labels_dl(self):
        return self.labels_dl

    def get_max_accuracy(self):
        return self.max_accuracy

    def get_min_accuracy(self):
        return self.min_accuracy

    def get_avg_accuracy(self):
        return self.avg_accuracy

    def get_max_accuracy_per_epoch(self):
        return self.max_accuracy_per_epoch

    def get_min_accuracy_per_epoch(self):
        return self.min_accuracy_per_epoch

    def get_accuracy_per_epoch(self):
        return self.accuracy_per_epoch


class InputGradients(Callback):
    def __init__(self, x_data, y_data, number_of_epochs):
        self.current_epoch = 0
        self.x = x_data
        self.y = y_data
        self.number_of_samples = len(x_data[0])
        self.number_of_epochs = number_of_epochs
        self.gradients = np.zeros((number_of_epochs, self.number_of_samples))
        self.gradients_sum = np.zeros(self.number_of_samples)

    def on_epoch_end(self, epoch, logs=None):
        print("Computing gradients...")

        input_trace = tf.Variable(self.x, dtype="float")

        with tf.GradientTape() as tape:
            tape.watch(input_trace)
            pred = self.model(input_trace)
            loss = tf.keras.losses.categorical_crossentropy(self.y, pred)

        grad = tape.gradient(loss, input_trace)

        print("Processing gradients...")
        input_gradients = np.zeros(self.number_of_samples)
        for i in range(len(self.x)):
            input_gradients += grad[i].numpy().reshape(self.number_of_samples)

        self.gradients[epoch] = input_gradients
        if np.max(self.gradients[epoch]) != 0:
            self.gradients_sum += np.abs(self.gradients[epoch] / np.max(self.gradients[epoch]))
        else:
            self.gradients_sum += np.abs(self.gradients[epoch])

        backend.clear_session()

    def grads(self):
        return np.abs(self.gradients_sum)

    def grads_epoch(self):
        for e in range(self.number_of_epochs):
            if np.max(self.gradients[e]) != 0:
                self.gradients[e] = np.abs(self.gradients[e] / np.max(self.gradients[e]))
            else:
                self.gradients[e] = np.abs(self.gradients[e])
        return self.gradients
