import timeit
from collections import OrderedDict
from collections import namedtuple
from itertools import product
from Optimizer import Optimizer

import numpy as np
from Algorithm import Algorithm

from search_space import param_decorator, SearchSpace


class GridSearch(Algorithm):
    def __init__(self, lb, ub):
        super(GridSearch, self).__init__(lb, ub)

        params = []
        for l, u in zip(lb, ub):
            params.append(np.arange(l, u))
        self.args = product(*params)

        self.best_params = None
        self.best_result = np.inf

    def run(self, eval_fn):
        for param in self.args:
            res = eval_fn(*param)
            if res < self.best_result:
                self.best_result = res
                self.best_params = param
        return self.best_params, self.best_result


class GSOptimizer(Optimizer):
    def __init__(self):
        super(GSOptimizer, self).__init__(GridSearch)
