import numpy as np
from numpy.random import rand
from collections import OrderedDict
import timeit
from collections import namedtuple

from search_space import param_decorator, SearchSpace
from utils import generate_sobol_sequences
from Optimizer import Optimizer
from Algorithm import Algorithm
from scoop import futures


class Particle:
    def __init__(self, position, phi1, phi2, lb, ub):
        self.phi1 = phi1
        self.phi2 = phi2
        self.lb = lb
        self.ub = ub

        self.velocity = np.ones(position.shape)
        self.position = position

        self.best_position = self.position
        self.best_fitness = np.inf

        self.evaluator = None

    def set_eval_fn(self, evaluator):
        self.evaluator = evaluator

    def get_position(self):
        return self.position

    def update_position(self, global_best_position):
        v1 = np.random.uniform(0, self.phi1, len(self.position)) * (self.best_position - self.position)
        v2 = np.random.uniform(0, self.phi2, len(self.position)) * (global_best_position - self.position)

        self.velocity = self.velocity + v1 + v2

        self.velocity = np.minimum(self.velocity, self.ub - self.position)
        self.velocity = np.maximum(self.velocity, self.lb - self.position)

        self.position = self.position + self.velocity
        return self.position

    def evaluate_fitness(self):
        fitness = self.evaluator.eval(*self.position)

        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position
        return fitness


class PSO(Algorithm):
    def __init__(self, lb, ub, num_generations, num_particles, phi1=1.5, phi2=2.0):
        super(PSO, self).__init__(lb, ub)

        self.num_generations = num_generations
        self.num_particles = num_particles

        positions = generate_sobol_sequences(num_particles, lb, ub)
        self.swarm = [Particle(pos, phi1, phi2, lb, ub) for pos in positions]

        self.best_fitness = np.inf
        self.best_position = None

    def run(self, evaluator):
        for particle in self.swarm:
            particle.set_eval_fn(evaluator)

        # evaluate particle
        fitnesses = list(futures.map(lambda p: p.evaluate_fitness(), self.swarm))

        self.best_fitness = np.min(fitnesses)
        self.best_position = self.swarm[np.argmin(fitnesses)].position

        for i in range(self.num_generations):

            # update particle
            for particle in self.swarm:
                particle.update_position(self.best_position)

            # evaluate particle
            fitnesses = list(futures.map(lambda p: p.evaluate_fitness(), self.swarm))
            self.best_fitness = np.min(fitnesses)
            self.best_position = self.swarm[np.argmin(fitnesses)].position

        return self.best_position, self.best_fitness


class PSOptimizer(Optimizer):
    def __init__(self, num_generations, num_particles, phi1=1.5, phi2=2.0):
        super(PSOptimizer, self).__init__(PSO, num_generations=num_generations, num_particles=num_particles, phi1=phi1,
                                          phi2=phi2)
