import numpy as np
import tensorflow as tf
import random
from gym.utils import seeding


def set_all_seeds(seed=1):
    rng, seed = seeding.np_random(1)

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return rng, seed


def get_obs_from_traj(trajectories):
    trajectories = np.array(trajectories)
    assert len(trajectories.shape) == 3

    def get_input_data(trajectory) -> list:
        return np.array(list(map(lambda step: step[0], trajectory)))

    obss = np.array(list(map(get_input_data, trajectories)))
    initial_shape = obss.shape

    return obss.reshape(-1, initial_shape[-1]), initial_shape
