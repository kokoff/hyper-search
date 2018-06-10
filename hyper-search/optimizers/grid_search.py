from itertools import product

import numpy as np

from Algorithm import Algorithm
from Optimizer import Optimizer
from utils import argmin


class GridSearch(Algorithm):
    def __init__(self, lb, ub, parallel):
        super(GridSearch, self).__init__(lb, ub, parallel)

        params = []
        for l, u in zip(lb, ub):
            params.append(np.arange(l, u))
        self.args = list(product(*params))

        self.best_params = None
        self.best_result = np.inf

    def run(self, evaluator):
        results = list(self.map(evaluator.eval, self.args))

        self.best_result = np.min(results)
        self.best_params = self.args[argmin(results)]
        return self.best_params, self.best_result


class GSOptimizer(Optimizer):
    def __init__(self):
        super(GSOptimizer, self).__init__(GridSearch)
