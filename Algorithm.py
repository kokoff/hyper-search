class Algorithm(object):

    def __init__(self, lb, ub, parallel):
        self.lb = lb
        self.ub = ub

        if parallel:
            from scoop import futures
            self.map = futures.map
        else:
            self.map = map

    def run(self, eval_fn):
        raise NotImplementedError
