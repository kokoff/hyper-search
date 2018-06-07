import timeit
from collections import OrderedDict
from collections import namedtuple
from itertools import product
from utils import generate_sobol_sequences
from Optimizer import Optimizer
from Algorithm import Algorithm
import numpy as np

from search_space import param_decorator, SearchSpace


class RandomSearch(Algorithm):
    def __init__(self, lb, ub, num_runs, sobol):
        super(RandomSearch, self).__init__(lb, ub)
        self.num_runs = num_runs
        self.sobol = sobol

        self.best_params = None
        self.best_result = np.inf

    def run(self, eval_fn):
        if self.sobol:
            params = generate_sobol_sequences(self.num_runs, self.lb, self.ub)
        else:
            params = [np.random.uniform(self.lb, self.ub) for i in range(self.num_runs)]

        for param in params:

            param = np.minimum(param, self.ub)
            param = np.maximum(param, self.lb)

            res = eval_fn(*param)
            if res < self.best_result:
                self.best_result = res
                self.best_params = param

        return self.best_params, self.best_result


class RSOptimizer(Optimizer):
    def __init__(self, num_runs, sobol=False):
        super(RSOptimizer, self).__init__(RandomSearch, num_runs=num_runs, sobol=sobol)


def func(x, y):
    print x, y
    return x ** 2 + y ** 2


def main():
    params = OrderedDict()
    params['x'] = (float, -10, 10)
    params['y'] = (float, -10, 10)

    opt = RSOptimizer(100, sobol=False)
    res = opt.optimize(func, params)
    print res.params
    print res.score
    print res.time


if __name__ == '__main__':
    main()
