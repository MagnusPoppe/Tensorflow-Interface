import numpy as np
import downing_code.tflowtools as TFT

class CaseManager():

    def __init__(self, case, minibatch_size, case_fraction, validation_fraction=0, test_fraction=0):
        self.casefunc = self.get_case_function(case, minibatch_size)

        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.training_fraction = 1 - (validation_fraction + test_fraction)

        self.generate_cases()

        if 0 < case_fraction < 1:
            self.total_case_fraction = case_fraction * len(self.cases)
        else: raise ValueError("case fraction needs to be between 0 and 1.")

        self.organize_cases()

    def get_case_function(self, case, minibatch_size):
        if case == "one-hot-bit": return lambda : TFT.gen_all_one_hot_cases(2 ** 4)

        # TODO: Implement the remaining datasets...
        if case == "vine quality": return None
        if case == "glass": return None
        if case == "yeast": return None
        if case == "mnist": return None

        # TODO: Implement the hacker's choice dataset
        if case == "hackers choice": return None


    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases[:int(self.total_case_fraction)])
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases

