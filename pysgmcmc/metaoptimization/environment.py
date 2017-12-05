import numpy as np
from gym import Env, spaces
from gym.utils import EzPickle
from keras import backend as K
from numpy import isfinite, isnan
from collections import namedtuple


SMALL_CONSTANT = 1e-4


class SamplerEnv(Env, EzPickle):
    metadata = {'render.modes': ["human"]}

    def __init__(self, sampler_constructor, param_factory, loss_function, initial_stepsize=1e-3, max_iterations=10 ** 4):
        self.sampler_constructor = sampler_constructor
        self.param_factory = param_factory
        self.loss_function = loss_function

        params = param_factory()
        self.sampler = sampler_constructor(
            params=params,
            loss=loss_function(params),
            lr=initial_stepsize
        )
        n_params = len(params)

        # assuming one stepsize per parameter
        self.n_stepsizes = len(self.sampler.learning_rates)

        self.viewer = None
        # observation space has stepsize(s)
        # XXX Anything else needed here?

        high = np.array([10.] * len(params))

        self.action_space = spaces.Discrete(n_params * 2)

        self.initial_stepsize = initial_stepsize

        self.stepsizes = np.asarray([self.initial_stepsize] * len(params))

        self.ACTION_LOOKUP = {}

        def incrementer(stepsizes, index, constant=SMALL_CONSTANT):
            stepsizes[index] += constant
            return stepsizes

        def decrementer(stepsizes, index, constant=SMALL_CONSTANT):
            stepsizes[index] -= constant
            return stepsizes

        # XXX: Add actions to increase/decrease constant?
        l = [
            [lambda stepsizes: incrementer(stepsizes, i),
             lambda stepsizes: decrementer(stepsizes, i)]
            for i in range(len(self.stepsizes))
        ]

        self.ACTION_LOOKUP = [el for sublist in l for el in sublist]

        assert len(self.ACTION_LOOKUP) == n_params * 2

        self.observation_space = spaces.Box(-high, high)

        self.max_iterations = max_iterations
        self.curr_iterations = 0

    def _step(self, action):
        fun = self.ACTION_LOOKUP[action]

        self.stepsizes = fun(self.stepsizes)
        results = self._apply_stepsizes(stepsizes=self.stepsizes)

        return results.observation, results.reward, results.episode_over, {}

    def _apply_stepsizes(self, stepsizes):
        self.curr_iterations += 1

        if isinstance(stepsizes, float):
            stepsizes = [stepsizes] * self.n_stepsizes
        else:
            stepsizes = list(stepsizes)

        # set stepsizes
        K.batch_set_value(list(zip(self.sampler.learning_rates, stepsizes)))

        # XXX Replace with sampler-based metrics, maybe use difference in metric values instead of absolute value
        cost, params = next(self.sampler)

        iterations_done = self.curr_iterations == self.max_iterations
        costs_diverged = (not isfinite(cost)) or isnan(cost)

        episode_over = iterations_done or costs_diverged

        if episode_over:
            self.curr_iterations = 0

        Step = namedtuple("Step", ["observation", "reward", "episode_over"])

        return Step(
            observation=stepsizes, reward=-cost, episode_over=episode_over
        )

    def _reset(self):
        # should reinitialize sampler with initial stepsize
        # and return our initial state again
        self.curr_iterations = 0
        params = self.param_factory()
        self.sampler = self.sampler_constructor(
            params=params,
            loss=self.loss_function(params),
            lr=self.initial_stepsize
        )

        return np.asarray([self.initial_stepsize] * len(params))

    def _render(self, mode="human", close=False):
        pass
