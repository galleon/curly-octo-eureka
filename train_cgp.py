import argparse
import cgp
from cgp.individual import IndividualSingleGenome
import functools
import itertools
from kaggle_environments.envs.mab.agents import random_agent
from kaggle_environments import make
import logging
import pickle

import random
from tqdm import tqdm, trange

from bas.submissionv3 import Agentv3 as BASAgent

# print(env.specification.configuration)
configuration = {
    "actTimeout": 0.25,
    "banditCount": 100,
    "decayRate": 0.97,
    "episodeSteps": 2000,
    "runTimeout": 1200,
    "sampleResolution": 100,
}

"""
How to clip variable in [-1, 1]
   var = max(-1.0, min(var, 1.0))
"""


class Exp(cgp.OperatorNode):
    """A node that calculates the exponential of its input."""

    _arity = 1
    _def_output = "math.exp(x_0)"
    _def_numpy_output = "np.exp(x_0)"
    _def_torch_output = "torch.exp(x_0)"
    _def_sympy_output = "exp(x_0)"


class ExpScaled(cgp.OperatorNode):
    """A node that calculates the exponential of its input. An adjustable
    parameter governs the scale.
    """

    _arity = 1
    _initial_values = {"<scale>": lambda: 1.0}
    _def_output = "math.exp(<scale> * x_0)"
    _def_numpy_output = "np.exp(<scale> * x_0)"
    _def_torch_output = "torch.exp(<scale> * x_0)"
    _def_sympy_output = "exp(<scale> * x_0)"


def display_msg(msg: str, no_skip: bool):
    if no_skip:
        print(msg)


def inner_objective(f, seed, n_runs_per_individual, debug=False):
    w = 0
    l = 0
    n = 0

    env = make("mab", debug=debug, configuration=configuration)

    for k in range(n_runs_per_individual):

        env.reset()

        ta = None

        def bas_agent(observation, configuration):
            global ta
            if observation.step == 0:
                ta = BASAgent(
                    configuration=configuration,
                    func=f,
                    debug=True,
                )

            action = ta.play(observation)

            return action

        r = env.run([bas_agent, random_agent])
        logging.debug("++++++++++++")
        logging.debug(r[-1][0])
        logging.debug("++++++++++++")

        if r[-1][0]["status"] == "INVALID":
            logging.debug("Unexpected error") 


        if r[-1][0]["reward"] > r[-1][1]["reward"]:
            w += 1
        elif r[-1][0]["reward"] < r[-1][1]["reward"]:
            l += 1
        else:
            n += 1

    display_msg(
        "fitness: {:.2f} ".format(1 + (w - l) / n_runs_per_individual),
        True,
    )

    return 1 + (w - l) / n_runs_per_individual


def objective(individual: IndividualSingleGenome, seed, n_runs_per_individual):

    print(individual.to_sympy())

    f = individual.to_numpy()

    # with open("callable.pkl", "wb") as fi:
    #     pickle.dump(f, fi)

    individual.fitness = inner_objective(f, seed, n_runs_per_individual)

    return individual


def main():
    objective_params = {"n_runs_per_individual": 10}

    population_params = {"n_parents": 10, "mutation_rate": 0.04, "seed": 42}

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 16,
        "n_rows": 1,
        "levels_back": None,
        "primitives": (
            cgp.Add,
            cgp.Sub,
            cgp.Mul,
            cgp.Div,
            cgp.ConstantFloat,
        ),
    }

    ea_params = {
        "n_offsprings": 4,
        "tournament_size": 1,
        "n_processes": 1,  # was 4
    }

    evolve_params = {"max_generations": 1000, "min_fitness": 0.0}

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    history = {"fitness_parents": [], "fitness_champion": []}

    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())
        history["fitness_champion"].append(pop.champion.fitness)

    obj = functools.partial(
        objective,
        seed=population_params["seed"],
        n_runs_per_individual=objective_params["n_runs_per_individual"],
    )

    cgp.evolve(
        pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback
    )

    display_msg(
        "History: {}".format(history),
        True,
    )
    display_msg(
        "Champion: {}".format(pop.champion.to_sympy()),
        True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
