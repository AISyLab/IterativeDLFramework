class DatasetParameters:

    def __init__(self):
        self.dataset_list = []

    def get_trace_set(self, trace_set_name):
        trace_list = self.get_trace_set_list()
        return trace_list[trace_set_name]

    def get_trace_set_list(self):
        parameters_cswap_arith = {
            "name": "cswap_arith",
            "data_length": 2,
            "first_sample": 0,
            "number_of_samples": 8000,
            "n_set1": 31875,
            "n_set2": 31875,
            "n_attack": 12750,
            "classes": 2,
            "epochs": 25,
            "mini-batch": 64
        }

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

        self.dataset_list = {
            "cswap_arith": parameters_cswap_arith,
            "cswap_pointer": parameters_cswap_pointer
        }

        return self.dataset_list
