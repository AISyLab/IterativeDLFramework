import numpy as np
import os
import h5py
import random
from tensorflow.keras import backend
from tensorflow.keras.utils import to_categorical
from commons.datasets import DatasetParameters
from commons.callbacks import CalculateAccuracyCurve25519, InputGradients, CalculateAccuracy
from commons.data_augmentation import DataAugmentation
from sklearn.utils import shuffle
from scipy import stats


class Framework:

    def __init__(self):
        self.directory = None
        self.dataset = None
        self.dataset_parameters = DatasetParameters()
        self.target_params = None
        self.callbacks = []
        self.model = None
        self.model_obj = None
        self.model_name = None
        self.hyper_parameters = []
        self.learning_rate = None
        self.optimizer = None
        self.da_active = False
        self.visualization_active = False

        self.z_score_mean = None
        self.z_score_std = None
        self.z_norm = False

        # Results
        self.accuracy_set1 = []
        self.accuracy_set2 = []
        self.max_trace_accuracy = []
        self.min_trace_accuracy = []
        self.avg_trace_accuracy = []
        self.input_gradients_epoch = None
        self.input_gradients_sum = None
        self.labels_dl = None

    def set_directory(self, directory):
        self.directory = directory

    def set_dataset(self, target):
        self.dataset = target
        self.dataset_parameters = DatasetParameters()
        self.target_params = self.dataset_parameters.get_trace_set(self.dataset)

    def set_database_name(self, database_name):
        self.database_name = database_name

    def sca_parameters(self):
        return self.dataset_parameters

    def set_znorm(self):
        self.z_norm = True

    def set_mini_batch(self, mini_batch):
        self.target_params["mini-batch"] = mini_batch

    def set_epochs(self, epochs):
        self.target_params["epochs"] = epochs

    def train_validation_sets(self):
        hf = h5py.File("{}{}.h5".format(self.directory, self.dataset), 'r')
        train_samples = np.array(hf.get('profiling_traces'))
        train_data = np.array(hf.get('profiling_data'))

        fs = self.target_params["first_sample"]
        ns = self.target_params["number_of_samples"]

        train_samples = train_samples[:, fs: fs + ns]
        if self.z_norm:
            self.create_z_score_norm(train_samples)
            self.apply_z_score_norm(train_samples)

        training_dataset_reshaped = train_samples.reshape((train_samples.shape[0], train_samples.shape[1], 1))

        x_train = training_dataset_reshaped[0:self.target_params["n_set1"]]
        x_val = training_dataset_reshaped[self.target_params["n_set1"]:self.target_params["n_set1"] + self.target_params["n_set2"]]
        y_train = train_data[0:self.target_params["n_set1"]]
        y_validation = train_data[self.target_params["n_set1"]:self.target_params["n_set1"] + self.target_params["n_set2"]]
        val_samples = train_samples[self.target_params["n_set1"]:self.target_params["n_set1"] + self.target_params["n_set2"]]

        return x_train, x_val, train_samples[0:self.target_params["n_set1"]], val_samples, y_train, y_validation

    def test_set(self):
        hf = h5py.File("{}{}.h5".format(self.directory, self.dataset), 'r')
        test_samples = np.array(hf.get('attacking_traces'))
        test_data = np.array(hf.get('attacking_data'))

        fs = self.target_params["first_sample"]
        ns = self.target_params["number_of_samples"]
        test_samples = test_samples[:, fs: fs + ns]

        if self.z_norm:
            self.apply_z_score_norm(test_samples)

        test_samples = test_samples[0:self.target_params["n_attack"]]
        y_test = test_data[0:self.target_params["n_attack"]]

        x_test = test_samples.reshape((test_samples.shape[0], test_samples.shape[1], 1))

        return x_test, test_samples, y_test

    def create_z_score_norm(self, dataset):
        self.z_score_mean = np.mean(dataset, axis=0)
        self.z_score_std = np.std(dataset, axis=0)

    def apply_z_score_norm(self, dataset):
        for s in range(self.target_params["number_of_samples"]):
            if self.z_score_std[s] == 0:
                self.z_score_std[s] = 1e-3
        for index in range(len(dataset)):
            dataset[index] = (dataset[index] - self.z_score_mean) / self.z_score_std

    def add_callback(self, callback):
        self.callbacks.append(callback)
        return self.callbacks

    def set_neural_network(self, model):
        self.model_name = model
        self.model_obj = model
        self.model = model(self.target_params["classes"], self.target_params["number_of_samples"])

    def get_model(self):
        return self.model

    def run(self):

        self.hyper_parameters = []

        x_train, x_val, train_samples, val_samples, train_data, validation_data = self.train_validation_sets()
        x_test, test_samples, test_data, = self.test_set()

        x_t = x_train
        x_v = x_val
        x_t1 = x_test

        print("-----------------------------------------------------------------------------------------------------------------------")
        print("Framework Phase 1")
        print("-----------------------------------------------------------------------------------------------------------------------")
        labels_from_pkc_data_train_ha = [0 if row[0] == 0 else 1 for row in train_data]
        labels_from_pkc_data_train_true = [0 if row[1] == 0 else 1 for row in train_data]
        labels_from_pkc_data_validation_ha = [0 if row[0] == 0 else 1 for row in validation_data]
        labels_from_pkc_data_validation_true = [0 if row[1] == 0 else 1 for row in validation_data]
        labels_from_pkc_data_test_ha = [0 if row[0] == 0 else 1 for row in test_data]
        labels_from_pkc_data_test_true = [0 if row[1] == 0 else 1 for row in test_data]

        y_train = to_categorical(labels_from_pkc_data_train_ha, num_classes=self.target_params["classes"])
        y_val = to_categorical(labels_from_pkc_data_validation_true, num_classes=self.target_params["classes"])

        correct_bits_training_set = 0
        for index in range(len(labels_from_pkc_data_train_true)):
            if labels_from_pkc_data_train_true[index] == labels_from_pkc_data_train_ha[index]:
                correct_bits_training_set += 1

        correct_bits_validation_set = 0
        for index in range(len(labels_from_pkc_data_validation_true)):
            if labels_from_pkc_data_validation_true[index] == labels_from_pkc_data_validation_ha[index]:
                correct_bits_validation_set += 1

        correct_bits_test_set = 0
        for index in range(len(labels_from_pkc_data_test_true)):
            if labels_from_pkc_data_test_true[index] == labels_from_pkc_data_test_ha[index]:
                correct_bits_test_set += 1

        print("Correct rate for training set: " + str(correct_bits_training_set / len(labels_from_pkc_data_train_true)))
        print("Correct rate for validation set: " + str(correct_bits_validation_set / len(labels_from_pkc_data_validation_true)))
        print("Correct rate for test set: " + str(correct_bits_test_set / len(labels_from_pkc_data_test_true)))

        callbacks_pkc_curve25519 = CalculateAccuracyCurve25519(x_t1, labels_from_pkc_data_test_true, labels_from_pkc_data_test_ha,
                                                               self.target_params["epochs"])

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        history = self.model.fit(
            x=x_t,
            y=y_train,
            batch_size=self.target_params["mini-batch"],
            verbose=1,
            epochs=self.target_params["epochs"],
            shuffle=True,
            validation_data=(x_v, y_val),
            callbacks=[callbacks_pkc_curve25519])

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        accuracy_dl = callbacks_pkc_curve25519.get_correct_dl()
        accuracy_ha = callbacks_pkc_curve25519.get_correct_ha()

        self.set_hyper_parameters()

        backend.clear_session()

        return accuracy_dl, callbacks_pkc_curve25519.get_max_accuracy_per_epoch()

    def run_iterative(self, n_iterations=50, data_augmentation=None, visualization=False, equal_size_traces=True):

        self.hyper_parameters = []

        self.learning_rate = backend.eval(self.model.optimizer.lr)
        self.optimizer = self.model.optimizer.__class__.__name__

        sca_data_augmentation = None
        if data_augmentation is not None:
            self.da_active = True
            sca_data_augmentation = DataAugmentation()

        if visualization:
            self.visualization_active = True
            self.input_gradients_epoch = np.zeros((n_iterations, self.target_params["epochs"], self.target_params["number_of_samples"]))
            self.input_gradients_sum = np.zeros((n_iterations, self.target_params["number_of_samples"]))

        x_set1, x_set2, set1_samples, set2_samples, set1_data, set2_data = self.train_validation_sets()
        x_attack, attack_samples, attack_data, = self.test_set()

        x_t1 = x_set1
        x_t2 = x_set2
        x_attack = x_attack

        set1_ha = [0 if row[0] == 0 else 1 for row in set1_data]
        set1_true = [0 if row[1] == 0 else 1 for row in set1_data]
        set2_ha = [0 if row[0] == 0 else 1 for row in set2_data]
        set2_true = [0 if row[1] == 0 else 1 for row in set2_data]
        test_ha = [0 if row[0] == 0 else 1 for row in attack_data]
        test_true = [0 if row[1] == 0 else 1 for row in attack_data]

        set1_ha_relabel = np.zeros(len(x_t1))
        set2_ha_relabel = np.zeros(len(x_t2))

        y_set1 = to_categorical(set1_ha, num_classes=self.target_params["classes"])
        y_set2 = to_categorical(set2_ha, num_classes=self.target_params["classes"])
        y_test = to_categorical(test_true, num_classes=self.target_params["classes"])

        sets_samples = np.zeros((len(x_t1) + len(x_t2), self.target_params["number_of_samples"]))
        sets_samples[0:len(x_t1)] = set1_samples
        sets_samples[len(x_t1):len(x_t1) + len(x_t2)] = set2_samples
        x_sets = sets_samples.reshape((sets_samples.shape[0], sets_samples.shape[1], 1))

        correct_bits_set1 = 0
        for index in range(len(set1_true)):
            if set1_true[index] == set1_ha[index]:
                correct_bits_set1 += 1

        correct_bits_set2 = 0
        for index in range(len(set2_true)):
            if set2_true[index] == set2_ha[index]:
                correct_bits_set2 += 1

        correct_bits_test_set = 0
        for index in range(len(test_true)):
            if test_true[index] == test_ha[index]:
                correct_bits_test_set += 1

        print("Correct scalar bits rate for set 1: " + str(correct_bits_set1 / len(set1_true)))
        print("Correct scalar bits rate for set 2: " + str(correct_bits_set2 / len(set2_true)))
        print("Correct scalar bits rate for test set: " + str(correct_bits_test_set / len(test_true)))

        self.labels_dl = np.zeros((n_iterations, self.target_params["epochs"], self.target_params["n_attack"]))

        max_accuracy = 0

        self.accuracy_set1 = []
        self.accuracy_set2 = []
        self.max_trace_accuracy = []
        self.min_trace_accuracy = []
        self.avg_trace_accuracy = []
        self.accuracy_set1.append(correct_bits_set1 / len(set1_true))
        self.accuracy_set2.append(correct_bits_set2 / len(set2_true))
        self.max_trace_accuracy.append(np.max([correct_bits_set1 / len(set1_true), correct_bits_set2 / len(set2_true)]))
        self.min_trace_accuracy.append(np.min([correct_bits_set1 / len(set1_true), correct_bits_set2 / len(set2_true)]))
        self.avg_trace_accuracy.append((self.max_trace_accuracy[0] + self.min_trace_accuracy[0]) / 2)

        callbacks_pkc_curve25519 = None
        callbacks_pkc = None

        for iteration_index in range(n_iterations):

            print("-----------------------------------------------------------------------------------------------------------------------")
            print("Framework Iteration {} - Phase 2".format(iteration_index))
            print("-----------------------------------------------------------------------------------------------------------------------")

            self.callbacks = []
            callback_input_gradients = None
            if self.visualization_active:
                callback_input_gradients = InputGradients(x_t1[0:2000], y_set1[0:2000], self.target_params["epochs"])
                self.add_callback(callback_input_gradients)
            if equal_size_traces:
                callbacks_pkc_curve25519 = CalculateAccuracyCurve25519(x_attack, test_true, test_ha, self.target_params["epochs"])
                callbacks = self.add_callback(callbacks_pkc_curve25519)
            else:
                callbacks_pkc = CalculateAccuracy(x_attack, test_true, test_ha, self.target_params["epochs"])
                callbacks = self.add_callback(callbacks_pkc)

            self.model = self.model_obj(self.target_params["classes"], self.target_params["number_of_samples"])
            if data_augmentation is not None:
                self.model.fit_generator(
                    generator=sca_data_augmentation.data_augmentation_shifts(set1_samples, y_set1, self.target_params),
                    steps_per_epoch=data_augmentation[0],
                    epochs=self.target_params["epochs"],
                    verbose=0,
                    validation_data=(x_attack, y_test),
                    validation_steps=1,
                    callbacks=callbacks)
            else:
                self.model.fit(
                    x=x_t1,
                    y=y_set1,
                    batch_size=self.target_params["mini-batch"],
                    verbose=0,
                    epochs=self.target_params["epochs"],
                    shuffle=True,
                    validation_data=(x_attack, y_test),
                    callbacks=callbacks)

            predictions = self.model.predict(x_t2)
            for index in range(len(x_t2)):
                set2_ha_relabel[index] = 0 if predictions[index][0] > 0.5 else 1
            if iteration_index > 0:
                y_set2 = to_categorical(set2_ha_relabel, num_classes=self.target_params["classes"])

            correct_bits_set2 = 0
            for index in range(len(set2_true)):
                if set2_true[index] == set2_ha_relabel[index]:
                    correct_bits_set2 += 1
            print("New correct rate for set 2: {}".format(correct_bits_set2 / len(set2_true)))
            self.accuracy_set2.append(correct_bits_set2 / len(set2_true))

            if equal_size_traces:
                max_accuracy_all_epochs = callbacks_pkc_curve25519.get_max_accuracy()
                min_accuracy_all_epochs = callbacks_pkc_curve25519.get_min_accuracy()
                avg_accuracy_set1 = callbacks_pkc_curve25519.get_avg_accuracy()
            else:
                max_accuracy_all_epochs = callbacks_pkc.get_max_accuracy()
                min_accuracy_all_epochs = callbacks_pkc.get_min_accuracy()
                avg_accuracy_set1 = callbacks_pkc.get_avg_accuracy()

            if max_accuracy_all_epochs > max_accuracy:
                max_accuracy = max_accuracy_all_epochs
            min_accuracy = min_accuracy_all_epochs

            print("\nMax Accuracy: {} (Iteration {})".format(max_accuracy, iteration_index))

            if self.visualization_active:
                self.input_gradients_epoch[iteration_index] = callback_input_gradients.grads_epoch()
                self.input_gradients_sum[iteration_index] = callback_input_gradients.grads()

            backend.clear_session()

            print("-----------------------------------------------------------------------------------------------------------------------")
            print("Framework Iteration {} - Phase 3".format(iteration_index))
            print("-----------------------------------------------------------------------------------------------------------------------")

            self.callbacks = []
            callback_input_gradients = None
            if self.visualization_active:
                callback_input_gradients = InputGradients(x_t2[0:2000], y_set2[0:2000], self.target_params["epochs"])
                self.add_callback(callback_input_gradients)

            if equal_size_traces:
                callbacks_pkc_curve25519 = CalculateAccuracyCurve25519(x_attack, test_true, test_ha, self.target_params["epochs"])
                callbacks = self.add_callback(callbacks_pkc_curve25519)
            else:
                callbacks_pkc = CalculateAccuracy(x_attack, test_true, test_ha, self.target_params["epochs"])
                callbacks = self.add_callback(callbacks_pkc)

            self.model = self.model_obj(self.target_params["classes"], self.target_params["number_of_samples"])
            if data_augmentation is not None:
                self.model.fit_generator(
                    generator=sca_data_augmentation.data_augmentation_shifts(set2_samples, y_set2, self.target_params),
                    steps_per_epoch=data_augmentation[0],
                    epochs=self.target_params["epochs"],
                    verbose=0,
                    validation_data=(x_attack, y_test),
                    validation_steps=1,
                    callbacks=callbacks)
            else:
                self.model.fit(
                    x=x_t2,
                    y=y_set2,
                    batch_size=self.target_params["mini-batch"],
                    verbose=0,
                    epochs=self.target_params["epochs"],
                    shuffle=True,
                    validation_data=(x_attack, y_test),
                    callbacks=callbacks)

            predictions = self.model.predict(x_t1)
            for index in range(len(x_t1)):
                set1_ha_relabel[index] = 0 if predictions[index][0] > 0.5 else 1
            if iteration_index > 0:
                y_set1 = to_categorical(set1_ha_relabel, num_classes=self.target_params["classes"])

            correct_bits_set1 = 0
            for index in range(len(set1_true)):
                if set1_true[index] == set1_ha_relabel[index]:
                    correct_bits_set1 += 1
            print("New correct rate for set 1: {}".format(correct_bits_set1 / len(set1_true)))
            self.accuracy_set1.append(correct_bits_set1 / len(set1_true))

            if equal_size_traces:
                max_accuracy_all_epochs = callbacks_pkc_curve25519.get_max_accuracy()
                min_accuracy_all_epochs = callbacks_pkc_curve25519.get_min_accuracy()
                avg_accuracy_set2 = callbacks_pkc_curve25519.get_avg_accuracy()
                self.labels_dl[iteration_index] = callbacks_pkc_curve25519.get_labels_dl()
            else:
                max_accuracy_all_epochs = callbacks_pkc.get_max_accuracy()
                min_accuracy_all_epochs = callbacks_pkc.get_min_accuracy()
                avg_accuracy_set2 = callbacks_pkc.get_avg_accuracy()
                self.labels_dl[iteration_index] = callbacks_pkc.get_labels_dl()

            if max_accuracy_all_epochs > max_accuracy:
                max_accuracy = max_accuracy_all_epochs
            if min_accuracy_all_epochs > min_accuracy:
                min_accuracy = min_accuracy_all_epochs

            print("\nMax Accuracy: {} (Iteration {})".format(max_accuracy, iteration_index))

            if self.visualization_active:
                self.input_gradients_epoch[iteration_index] = callback_input_gradients.grads_epoch()
                self.input_gradients_sum[iteration_index] = callback_input_gradients.grads()

            backend.clear_session()

            self.max_trace_accuracy.append(max_accuracy)
            self.min_trace_accuracy.append(min_accuracy)
            self.avg_trace_accuracy.append((avg_accuracy_set1 + avg_accuracy_set2) / 2)

            print("-----------------------------------------------------------------------------------------------------------------------")
            print("Framework Iteration {} - Phase 4".format(iteration_index))
            print("-----------------------------------------------------------------------------------------------------------------------")

            print("Joining set1 and set 2...")
            data_relabel = np.zeros(len(x_t1) + len(x_t2))
            data_relabel[0:len(x_t1)] = set1_ha_relabel
            data_relabel[len(x_t1):len(x_t1) + len(x_t2)] = set2_ha_relabel

            data_true = np.zeros(len(x_t1) + len(x_t2))
            data_true[0:len(x_t1)] = set1_true
            data_true[len(x_t1):len(x_t1) + len(x_t2)] = set2_true

            print("Shuffling set1 + set 2...")
            rnd_state = random.randint(0, 100000)
            sets_samples_shuffle, relabel_shuffle, true_shuffle = shuffle(sets_samples, data_relabel, data_true, random_state=rnd_state)

            print("Splitting into set1 and set 2...\n")
            set1_ha = relabel_shuffle[0: len(x_t1)]
            set2_ha = relabel_shuffle[len(x_t1): len(x_t1) + len(x_t2)]
            set1_true = true_shuffle[0: len(x_t1)]
            set2_true = true_shuffle[len(x_t1): len(x_t1) + len(x_t2)]

            # return shuffled sets 1 and 2
            y_set1 = to_categorical(set1_ha, num_classes=self.target_params["classes"])
            y_set2 = to_categorical(set2_ha, num_classes=self.target_params["classes"])

            set1_samples = sets_samples_shuffle[0:len(x_t1)]
            set2_samples = sets_samples_shuffle[len(x_t1):len(x_t1) + len(x_t2)]

            sets_samples = sets_samples_shuffle
            x_sets = sets_samples_shuffle.reshape((sets_samples_shuffle.shape[0], sets_samples_shuffle.shape[1], 1))
            x_set1 = x_sets[0:len(x_t1)]
            x_set2 = x_sets[len(x_t1):len(x_t1) + len(x_t2)]

            x_t1 = x_set1
            x_t2 = x_set2

        self.set_hyper_parameters()

    def run_ttest(self, labels, samples):

        n1 = sum(labels)
        n0 = len(labels) - n1

        set1 = np.zeros((int(n1), len(samples[0])))
        set0 = np.zeros((int(n0), len(samples[0])))

        n0_count = 0
        n1_count = 0
        for i in range(int(n0 + n1)):
            if labels[i] == 0:
                set0[n0_count] = samples[i]
                n0_count += 1
            else:
                set1[n1_count] = samples[i]
                n1_count += 1

        return stats.ttest_ind(set0, set1)[0]

    def __set_hyper_parameters(self):
        self.hyper_parameters.append({
            "mini_batch": self.target_params["mini-batch"],
            "epochs": self.target_params["epochs"],
            "learning_rate": float(self.learning_rate),
            "optimizer": str(self.optimizer),
            "set1": self.target_params["n_set1"],
            "set2": self.target_params["n_set2"],
            "attack_set": self.target_params["n_attack"]
        })

    def set_hyper_parameters(self):
        self.__set_hyper_parameters()

    def get_hyper_parameters(self):
        return self.hyper_parameters

    def get_input_gradients_epochs(self):
        return self.input_gradients_epoch

    def get_input_gradients_sum(self):
        return self.input_gradients_sum

    def get_accuracy_set1(self):
        return self.accuracy_set1

    def get_accuracy_se2(self):
        return self.accuracy_set2

    def get_max_trace_accuracy(self):
        return self.max_trace_accuracy

    def get_min_trace_accuracy(self):
        return self.min_trace_accuracy

    def get_avg_trace_accuracy(self):
        return self.avg_trace_accuracy

    def get_labels_dl(self):
        return self.labels_dl
