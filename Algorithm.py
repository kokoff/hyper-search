class Algorithm(object):

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def run(self, eval_fn):
        raise NotImplementedError
