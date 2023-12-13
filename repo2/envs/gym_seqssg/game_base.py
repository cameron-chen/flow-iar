#Numba supported seqssg env

import math

import numpy as np
from numba import b1, float32, float64, int32, typed, types, uint32
from numba.experimental import jitclass
from numba.typed import Dict

from utils.util import (one_hot, one_hot_w_padding, sample_cartesian_product,
                        softmax)

# import numba


class HelperClass:
    def __init__(
        self, 
        num_rsc: int,
        num_tgt: int,
    ):

        self.num_rsc = num_rsc
        self.num_tgt = num_tgt

    #4
    def decode_tgt_fea_int(self, obs: np.ndarray): 
        """
        Decode the mixed integer matrix into a dictionary of target features.

        Args:
            obs: a mixed integer matrix of shape (num_tgt, 1+num_rsc+1+num_tgt).

        Returns:
            A dictionary of target features, with keys as target indices and values
                as dictionaries of features.
        """
        assert obs.ndim == 2
        assert obs.shape[1] == 2 + self.num_rsc + self.num_tgt

        tgt_fea_mat = obs[:, : -self.num_tgt]
        tgt_fea = {i: {'def_st': 0, 'def_rsc': [], 'atk': 0} for i in range(self.num_tgt)}
        rsc_fea = {i: 0 for i in range(self.num_rsc)}

        for k, v in tgt_fea.items():
            v['def_st'] = tgt_fea_mat[k, 0].item()
            def_rsc_np = tgt_fea_mat[k, 1: -1]
            v['def_rsc'] = def_rsc_np[def_rsc_np < self.num_rsc].tolist()
            v['atk'] = tgt_fea_mat[k, -1].item()

            for rsc in v['def_rsc']:
                rsc_fea[rsc] = k
        
        print("decode tgt fea", tgt_fea)
        print("decode rsc fea", tgt_fea)
        return tgt_fea, rsc_fea


kv_ty = (types.int64, types.int64)

spec = [
    ("num_rsc", int32),
    ("num_tgt", int32),
    ("rsc_fea", types.DictType(*kv_ty)),
    ("cur_rsc_fea", types.DictType(*kv_ty)),
    ("asgmt", int32[:]),
    ("def_act", int32[:]),
    ("available_asgmt", int32[:, :]),
    ("def_constraints", int32[:, :]),
    ("numba_def_rsc", types.ListType(types.ListType(int32))),
    ("adj", int32[:, :]),
    ("cost", float64[:, :]),
]


@jitclass(spec)
class SeqSSGBase:
    def __init__(
        self, 
        num_rsc: int,
        num_tgt: int,
    ):
        self.num_rsc = num_rsc
        self.num_tgt = num_tgt

    #9
    def check_adj(self, asgmt: np.ndarray, available_move: np.ndarray, rsc_fea, cur_rsc_fea:Dict=None) -> bool:
        """Check if the assignment satisfies the adjacency constraints."""
        # assert asgmt.ndim == 1 
        # assert asgmt.dtype in [np.int32, np.int64] #removed since defined dtypes using numba
        # if cur_rsc_fea is None: cur_rsc_fea = copy.deepcopy(self.rsc_fea)  #copy not in numba
        if cur_rsc_fea is None: cur_rsc_fea = rsc_fea
        
        move_pair = []
        for i, tgt in enumerate(asgmt):
            if cur_rsc_fea[i] != tgt: # If the resource is moved (not check the fixed resource)
                move_pair.append((cur_rsc_fea[i], tgt))
        
        return np.all(np.array([available_move[pair] for pair in move_pair]))

    #10
    def check_asgmt_cstr(self, asgmt: np.ndarray, available_asgmt: np.ndarray, def_constraints: np.ndarray) -> bool:
        """Check if the assignment satisfies the resource constraints."""

        pair_locs = [(asgmt[p1], asgmt[p2]) for (p1,p2) in def_constraints]
        return np.all(np.array([available_asgmt[p_l] for p_l in pair_locs]))

   
    #14   
    def _verify_adj_and_cost(
        self, 
        adj: np.ndarray,
        cost: np.ndarray=None, 
    ):
        """Verify the adjacency matrix and the cost matrix."""
        # Adjacency matrix
        assert adj.ndim == 2
        assert np.all(adj >= 0)
        assert np.all(adj <= 1)
        
        # Cost matrix
        if cost is not None:
            assert cost.shape == adj.shape
            assert cost.min() == -1.0
            assert np.equal(np.where(cost > 0, 1, 0), adj).all()



    





    

