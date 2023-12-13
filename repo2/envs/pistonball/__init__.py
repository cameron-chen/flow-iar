import gym
import numpy as np

from .util import PistonToGymWrapper, PistonWithCstrToGymWrapper

id_attr = {
    "Pistonball-v0": {
        'entry_point':PistonToGymWrapper,
        'n_pistons': 5,
    },
    "Pistonball-v1": {
        'entry_point':PistonToGymWrapper,
        'n_pistons': 8,
    },
    "Pistonball-v2": {
        'entry_point':PistonToGymWrapper,
        'n_pistons': 12,
    },
    "PistonballNum10-v1": {
        'entry_point':PistonToGymWrapper,
        'n_pistons': 10,
    },
    "PistonballNum11-v1": {
        'entry_point':PistonToGymWrapper,
        'n_pistons': 11,
    },
    "PistonballCstr-v0": {
        'entry_point':PistonWithCstrToGymWrapper,
        'n_pistons': 5,
        'obs_type': 'global_processed',
    },
    "PistonballCstr-v1": {
        'entry_point':PistonWithCstrToGymWrapper,
        'idx_pistons_cstr':np.arange(4),
        'n_pistons': 8,
        'obs_type': 'global_processed',
    },
}

for env_id, attr in id_attr.items():
    entry_point = attr.pop('entry_point')
    gym.register(id=env_id, entry_point=entry_point, kwargs=attr)