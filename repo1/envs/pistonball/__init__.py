import gym

from .util import PistonToGymWrapper

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
}

for env_id, attr in id_attr.items():
    entry_point = attr.pop('entry_point')
    gym.register(id=env_id, entry_point=entry_point, kwargs=attr)