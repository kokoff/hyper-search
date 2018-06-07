from particle_swarm import PSOptimizer
from random_serch import RSOptimizer
from grid_search import GSOptimizer
from search_space import Variable, Choice


def func(x, y):
    return x ** 2 + y ** 2


def pso():
    params = {}
    params['x'] = Variable(-100000, 100000, float)
    params['y'] = Variable(-100000, 100000, float)

    opt = PSOptimizer(100, 100)
    res = opt.optimize(func, params)
    print res.params
    print res.score
    print res.time


def grid_search():
    params = {}
    params['x'] = Variable(0, 10, int)
    params['y'] = Choice(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    opt = GSOptimizer()
    res = opt.optimize(func, params)

    print res


def random_search():
    params = {}
    params['x'] = Variable(0, 100, float)
    params['y'] = Choice(Variable(10, 100, int), Variable(0, 10, float))

    opt = RSOptimizer(100)
    res = opt.optimize(func, params)

    print res


if __name__ == '__main__':
    # grid_search()
    # random_search()
    pso()
