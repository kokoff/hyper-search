import numpy as np

from Algorithm import Algorithm
from Optimizer import Optimizer
from utils import generate_sobol_sequences, argmin


class RandomSearch(Algorithm):
    def __init__(self, lb, ub, parallel, num_runs, init):
        super(RandomSearch, self).__init__(lb, ub, parallel)
        self.num_runs = num_runs

        self.params = self.initialize(lb, ub, num_runs, init)
        self.params = [np.minimum(param, self.ub) for param in self.params]
        self.params = [np.maximum(param, self.lb) for param in self.params]

        self.best_params = None
        self.best_result = np.inf

    def initialize(self, lb, ub, num_runs, init):
        if init == 'sobol':
            return generate_sobol_sequences(num_runs, lb, ub)
        elif init == 'uniform':
            return [np.random.uniform(lb, ub) for _ in range(num_runs)]
        elif init == 'normal':
            means = np.mean([lb, ub], axis=0)
            stds = (means - lb) / 3
            return [np.random.normal(means, stds) for _ in range(num_runs)]

    def run(self, evaluator):

        results = list(self.map(evaluator.eval, self.params))

        self.best_result = np.min(results)
        self.best_params = self.params[argmin(results)]

        return self.best_params, self.best_result


class RSOptimizer(Optimizer):
    def __init__(self, num_runs, init='uniform'):
        super(RSOptimizer, self).__init__(RandomSearch, num_runs=num_runs, init=init)
