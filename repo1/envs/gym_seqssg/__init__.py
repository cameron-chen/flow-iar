import json

import gym
import numpy as np

# from .base import SeqSSG          #no-numba
from .game_env_base import SeqSSG  # numba-supported

# ============ Toy environments - V0 ============
n_rsc = 3
n_tgt = 5 
payoff_matrix = np.array([
    [1,   -1,   1.5, -1.2], 
    [0.9, -0.9, 0.7, -0.5],
    [0.8, -0.8, 0.5, -0.45],
    [1,   -1,   0.8, -0.7],
    [1.1, -1.1, 1.3, -1]
])
adj_matrix = np.array([
    [0,0,0,0,1],
    [0,0,1,1,0],
    [0,1,0,0,0],
    [0,1,0,0,1],
    [1,0,0,1,0]
])
cost_matrix = np.array([
    [ 0.  , -1.  , -1.  , -1.  ,  0.46],
    [-1.  ,  0.  ,  0.49,  0.38, -1.  ],
    [-1.  ,  0.49,  0.  , -1.  , -1.  ],
    [-1.  ,  0.38, -1.  ,  0.  ,  0.4 ],
    [ 0.46, -1.  , -1.  ,  0.4 ,  0.  ]
])
def_constraints = [(1,2)]
id_attr = {
    "SeqSSG-no-cstr-v0": {
        "has_constraint": False,
        "has_cost": False,
    }, 
    # Rationale for SeqSSG-v0:
    #   req_hop = 3: relax adjacency constraints
    #   req_dist = 1: enforce assignment constraints
    #   This makes sure that feasible actions exist and req_dist provides necessary
    #   constraints such that the problem would not be too easy.
    "SeqSSG-v0": { 
        "has_constraint": True,
        "has_cost": True,
        "req_hop": 3,
        "req_dist": 1,
        "inv_pen": -7.0 # Default value for version 0
    }
}

for env_id, attr in id_attr.items():
    kwargs = {
        "payoff_matrix": payoff_matrix,
        "adj_matrix": adj_matrix,
        "cost_matrix": cost_matrix,
        "def_constraints": def_constraints,
        "num_rsc": n_rsc,
        "num_tgt": n_tgt,
    }
    kwargs.update(attr)

    gym.register(id=env_id, entry_point=SeqSSG,
                max_episode_steps=100, kwargs=kwargs)

# ============ Toy environments - V1 ============
n_rsc = 3
n_tgt = 6
payoff_matrix = np.array([
    [1,   -1,   1.5, -1.2], 
    [0.9, -0.9, 0.7, -0.5],
    [0.8, -0.8, 0.5, -0.45],
    [1,   -1,   0.8, -0.7],
    [1.1, -1.1, 1.3, -1],
    [1.2, -1.2, 1.4, -1.1]
])
adj_matrix = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
])
cost_matrix = np.array([
    [ 0.  ,  0.1 ,  0.17, -1.  , -1.  , -1.  ],
    [ 0.1 ,  0.  ,  0.2 , -1.  ,  0.07,  0.12],
    [ 0.17,  0.2 ,  0.  ,  0.12, -1.  , -1.  ],
    [-1.  , -1.  ,  0.12,  0.  , -1.  , -1.  ],
    [-1.  ,  0.07, -1.  , -1.  ,  0.  , -1.  ],
    [-1.  ,  0.12, -1.  , -1.  , -1.  ,  0.  ]
])
def_constraints = [(1,2)]
attr_for_testing = {
    'init_state': {
        0: np.array([3, 4, 1]),
        1: np.array([0, 3, 5]),
        2: np.array([2, 0, 4]),
        3: np.array([1, 3, 2]),
        4: np.array([1, 5, 2]),
        5: np.array([1, 3, 4]),
        6: np.array([4, 1, 2]),
        7: np.array([0, 3, 2]),
        8: np.array([4, 1, 5]),
        9: np.array([1, 2, 0]),
    }
}
id_attr = {
    "SeqSSG-no-cstr-v1": {
        "has_constraint": False,
        "has_cost": False,
    }, 
    "SeqSSG-v1": { 
        "has_constraint": True,
        "has_cost": True,
        "req_hop": 3,
        "req_dist": 1,
    },
    "SeqSSG-payoff-v1": { 
        "has_constraint": True,
        "has_cost": True,
        "req_hop": 3,
        "req_dist": 1,
        "att_param": {
            'dist_payoff':np.array([0.7, 0.2, 0.1]),
            'idx_tgt':np.array([
                [0, 1, 2, 3, 4, 5],
                [3, 5, 4, 2, 0, 1],
                [5, 4, 2, 1, 0, 3],
            ]),}
    },
    "SeqSSG-0-rew-v1": { 
        "has_constraint": True,
        "has_cost": True,
        "req_hop": 3,
        "req_dist": 1,
        "inv_pen": 0.0
    }
}

for env_id, attr in id_attr.items():
    kwargs = {
        "payoff_matrix": payoff_matrix,
        "adj_matrix": adj_matrix,
        "cost_matrix": cost_matrix,
        "def_constraints": def_constraints,
        "num_rsc": n_rsc,
        "num_tgt": n_tgt,
        "attr_for_testing": attr_for_testing,
    }
    kwargs.update(attr)

    gym.register(id=env_id, entry_point=SeqSSG,
                max_episode_steps=100, kwargs=kwargs)
            
# ============ Large-scale environments ============
for i in range(2, 11): 
    env_id = f"SeqSSG-v{i}"

    # load attributes from file
    with open(f"envs/gym_seqssg/config/seqssg_v{i}.json", "r") as f:
        attr = json.load(f)
    
    kwargs = {
        "payoff_matrix": np.array(attr["payoff_matrix"]),
        "adj_matrix": np.array(attr["adj_matrix"], dtype=np.int),
        "cost_matrix": np.array(attr["cost_matrix"]),
        "def_constraints": attr["def_constraints"],
        "num_rsc": attr["num_rsc"],
        "num_tgt": attr["num_tgt"],
        "has_constraint": True, 
        "has_cost": True,
        "req_hop": attr.get('req_hop', 3),
        "req_dist": attr.get("req_dist", 1),
    }

    gym.register(id=env_id, entry_point=SeqSSG,
                max_episode_steps=100, kwargs=kwargs)

for i in range(2, 11): 
    env_id = f"SeqSSG-payoff-v{i}"

    # load attributes from file
    with open(f"envs/gym_seqssg/config/seqssg_v{i}.json", "r") as f:
        attr = json.load(f)
    
    if attr.get("att_param", None) is None:
        continue

    kwargs = {
        "payoff_matrix": np.array(attr["payoff_matrix"]),
        "adj_matrix": np.array(attr["adj_matrix"], dtype=np.int),
        "cost_matrix": np.array(attr["cost_matrix"]),
        "def_constraints": attr["def_constraints"],
        "num_rsc": attr["num_rsc"],
        "num_tgt": attr["num_tgt"],
        "has_constraint": True, 
        "has_cost": True,
        "req_hop": attr.get('req_hop', 3),
        "req_dist": attr.get("req_dist", 1),
        "att_param": attr["att_param"],
        "fixed_assign": attr.get("fixed_assign", None),
        "attr_for_testing": attr.get("attr_for_testing", None),
    }

    gym.register(id=env_id, entry_point=SeqSSG,
                max_episode_steps=100, kwargs=kwargs)

for i in range(2, 11):
    env_id = f"SeqSSG-no-cstr-v{i}"

    # load attributes from file
    with open(f"envs/gym_seqssg/config/seqssg_v{i}.json", "r") as f:
        attr = json.load(f)
    
    kwargs = {
        "payoff_matrix": np.array(attr["payoff_matrix"]),
        "adj_matrix": np.array(attr["adj_matrix"], dtype=np.int),
        "cost_matrix": np.array(attr["cost_matrix"]),
        "def_constraints": attr["def_constraints"],
        "num_rsc": attr["num_rsc"],
        "num_tgt": attr["num_tgt"],
        "has_constraint": False, 
        "has_cost": False,
        "att_param": attr.get("att_param", None),
        "attr_for_testing": attr.get("attr_for_testing", None),
    }

    gym.register(id=env_id, entry_point=SeqSSG,
                max_episode_steps=100, kwargs=kwargs)