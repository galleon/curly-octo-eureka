from kaggle_environments.envs.mab.mab import Observation, Configuration
from collections import defaultdict
import copy
import logging
from math import erf, exp, log, sqrt
import numpy as np
import operator
import os
import pickle
import random
from typing import Callable, Dict, List

configuration = {
    "actTimeout": 0.25,
    "banditCount": 100,
    "decayRate": 0.97,
    "episodeSteps": 2000,
    "runTimeout": 1200,
    "sampleResolution": 100,
}

logging.basicConfig(filename="submissionv3.log", level=logging.DEBUG)
logging.debug("----------------------------------------------------------------------")


def display_msg(msg: str, no_skip: bool):
    if no_skip:
        print(msg)


class Agentv3:
    epsilon = 1e-32

    def __init__(
        self,
        configuration: Configuration,
        func,
        debug: bool = False,
    ):
        """
        k: the number of bandits
        n: number of pulls per arm
        """
        self.k = configuration["banditCount"]
        self.decay = configuration["decayRate"]
        self.episode = configuration["episodeSteps"]
        # A bandit can return between 0 and sampleResolution included
        # The code use random.randint(0, configuration.sample_resolution)
        self.resolution = configuration["sampleResolution"]
        #
        self.total_reward = 0
        self.step = 0
        self.debug = debug
        #
        self.proba = [
            np.ones(shape=(self.resolution + 1,)) / (self.resolution + 1)
            for _ in range(self.k)
        ]
        #
        # In this array, we store for each agent the time step at which the bandit was pulled
        self.selections = [[[] for k in range(self.k)], [[] for k in range(self.k)]]
        self.pulls = []
        #
        self.f = func
        #
        self.set_seed()

        self.played_by_adversary = dict()
        logging.debug("Agentv3 created")

    def set_seed(self, my_seed=42):
        os.environ["PYTHONHASHSEED"] = str(my_seed)
        random.seed(my_seed)

    def select(self):
        #
        # Analysis of what the opponent has played so far
        #
        logging.debug("    Entering select()")

        threshold1 = 0.9

        safe_prob_indices = [
            np.sum((np.cumsum(self.proba[k]) <= threshold1).astype(int))
            for k in range(self.k)
        ]
        #
        m = np.array(
            [
                np.dot(
                    np.array(
                        [(t / self.resolution) for t in range(self.resolution + 1)]
                    ),
                    self.proba[k],
                )
                for k in range(self.k)
            ]
        )
        # compute sigma for each bandit
        s = np.array(
            [
                sqrt(
                    np.dot(
                        np.array(
                            [
                                (t / self.resolution) ** 2
                                for t in range(self.resolution + 1)
                            ]
                        ),
                        self.proba[k],
                    )
                    - m[k] ** 2
                )
                for k in range(self.k)
            ]
        )
        #
        # display_msg("values1: {}".format(values1), self.debug)

        logging.debug("---- Inputs ----")
        # logging.debug(m.__class__)
        # logging.debug(m[:5])
        # logging.debug(s[:5])
        # logging.debug(self.f)
        # logging.debug(dir(self.f))

        u = np.vstack([m, s]).T
        logging.debug(u.shape)

        v = self.f(u)

        logging.debug("---- Outputs ----")
        logging.debug(vv.__class__)
        logging.debug(vv.shape)
        logging.debug(dir(vv))
        logging.debug("   {}".format(vv.argmax()))
        logging.degug("- {} -".format(vv[:5]))
        # logging.degug("- {} -".format(int(np.argmax(v))))

        action = int(v.argmax())
        logging.debug("   {}".format(action))
        logging.debug("    Exiting select()")

        return int(v.argmax())

    def update(self, observation):
        logging.debug("    Entering update()")
        reward = observation["reward"] - self.total_reward
        self.step = observation["step"]

        logging.debug("<{}>".format(observation["lastActions"]))

        mla = None
        hla = None
        for agent, k in enumerate(observation["lastActions"]):
            # nk is the number of times bandit k was pulled
            if agent == observation["agentIndex"]:
                # This is me ;-)
                p = np.array(
                    [
                        t / self.resolution * self.decay ** nk
                        for t in range(self.resolution + 1)
                    ]
                )
                if reward > 0:  # success
                    p = np.multiply(p, self.proba[k])
                else:  # failure
                    p = np.multiply(1 - p, self.proba[k])

                self.proba[k] = p / np.sum(p)
                #
                mla = k
            else:
                hla = k

            self.selections[agent][k].append(observation["step"])

        self.total_reward += reward

        logging.debug("    Exiting update()")

    def play(self, observation):
        logging.debug("Entering play()")

        self.update(observation)

        action = self.select()

        logging.debug("- {} -".format(action))

        self.pulls.append(action)

        logging.debug(
            "[{:4d}] Elected action: {}[{}]".format(self.step, action, len(self.proba))
        )
        logging.debug("Exiting play()")
        action = 1

        return action


class MyFunction:
    def __init__(self, name="default"):
        self.name = name

    def __call__(self, x):
        return [np.add(x[0], x[1])]


agent = None


def bas_agent(observation, configuration=configuration):
    global agent
    if observation.step == 0:
        agent = Agentv3(
            configuration,
            func=None,
            debug=True,
        )

    action = agent.play(observation)

    return action
