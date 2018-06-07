from search_space import SearchSpace, param_decorator, Evaluator
import timeit
from collections import namedtuple


class Optimizer(object):
    def __init__(self, algorithm, *args, **kwargs):
        self.algorithm = algorithm
        self.args = args
        self.kwargs = kwargs

    def optimize(self, run_f, params):
        search_tree = SearchSpace(params)

        lb = search_tree.get_lb()
        ub = search_tree.get_ub()
        f = Evaluator(run_f, search_tree)

        algorithm = self.algorithm(lb, ub, *self.args, **self.kwargs)

        start = timeit.default_timer()
        best_params, score = algorithm.run(f)
        end = timeit.default_timer() - start

        best_params = search_tree.transform(best_params)
        # Result = namedtuple('Result', ['params', 'score', 'time'])

        return Result(best_params, score, end)


class Result(object):
    def __init__(self, params, score, time):
        self.params = params
        self.score = score
        self.time = time

    def __str__(self):
        string = ''
        string += 'Result: params=' + str(self.params) + ', score=' + str(self.score) + ', time=' + str(self.time)
        return string
