import numpy as np


class InputLayer:

    def __init__(self, n_att):
        self.out = np.random.uniform(size=[n_att, 1])
        # self.out = np.array(n_att)


    def init_input(self, input_given):
        self.out = input_given
