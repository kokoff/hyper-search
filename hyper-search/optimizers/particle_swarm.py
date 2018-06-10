import numpy as np

from Algorithm import Algorithm
from Optimizer import Optimizer
from utils import generate_sobol_sequences
from utils import argmin


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
        fitness = self.evaluator.eval(self.position)

        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position
        return fitness


class PSO(Algorithm):
    def __init__(self, lb, ub, parallel, num_generations, num_particles, phi1, phi2, init):
        super(PSO, self).__init__(lb, ub, parallel)

        self.num_generations = num_generations
        self.num_particles = num_particles

        positions = self.initialize(num_particles, lb, ub, init)
        self.swarm = [Particle(pos, phi1, phi2, lb, ub) for pos in positions]

        self.best_fitness = np.inf
        self.best_position = None

    def initialize(self, num_particles, lb, ub, init):
        if init == 'sobol':
            return generate_sobol_sequences(num_particles, lb, ub)
        elif init == 'uniform':
            return [np.random.uniform(lb, ub) for _ in range(num_particles)]
        elif init == 'normal':
            means = np.mean([lb, ub], axis=0)
            stds = (means - lb) / 3
            return [np.random.normal(means, stds) for _ in range(num_particles)]

    def run(self, evaluator):
        for particle in self.swarm:
            particle.set_eval_fn(evaluator)

        for _ in range(self.num_generations + 1):
            # evaluate particles
            fitnesses = list(self.map(lambda p: p.evaluate_fitness(), self.swarm))

            # current best particle
            current_best_fitness = min(fitnesses)
            current_best_position = self.swarm[argmin(fitnesses)].position

            # overall best particle
            if current_best_fitness <= self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_position = current_best_position

            # update particles
            for particle in self.swarm:
                particle.update_position(self.best_position)

        return self.best_position, self.best_fitness


class PSOptimizer(Optimizer):
    def __init__(self, num_generations, num_particles, phi1=1.5, phi2=2.0, init='sobol'):
        super(PSOptimizer, self).__init__(PSO, num_generations=num_generations, num_particles=num_particles, phi1=phi1,
                                          phi2=phi2, init=init)
