#Numba supported seqssg env

import copy
import warnings
from typing import List, Optional, Tuple

import gym
import networkx as nx
import numpy as np
from einops import rearrange, repeat
from gym import spaces
from numba import njit, typed, typeof, types

from envs.gym_seqssg.game_base import SeqSSGBase
from utils.util import (one_hot, one_hot_w_padding, sample_cartesian_product,
                        softmax)


class SeqSSG(gym.Env):
    def __init__(
        self, 
        *,
        payoff_matrix: np.ndarray, 
        adj_matrix: np.ndarray, 
        cost_matrix: np.ndarray, 
        num_rsc: int,
        num_tgt: int,
        init_state: np.ndarray=None,
        def_constraints: List[Tuple]=[], 
        atk_eps: float=0.05,
        inv_pen: float=None,
        req_hop: int=1,
        req_dist: int=1,
        edge_matrix_type: str='adj',
        has_constraint: bool=True,
        has_cost: bool=True,
        att_param: dict=None,
        fixed_assign: dict=None,
        attr_for_testing: dict=None,
    ):
        """Suqential SSG environment. 

        Args:
            payoff_matrix (np.ndarray): (num_tgt, 4) array, payoff matrix. 
                Vars: (def_rew, def_plt, atk_rew, atk_plt)
            adj_matrix (np.ndarray): (num_tgt, num_tgt) array, binary adjacency matrix.
            cost_matrix (np.ndarray): (num_tgt, num_tgt) array, cost matrix. Assume that 
                1. the cost of moving from a target to itself is 0.
                2. the cost of moving from a target to another target is positive.
                3. the cost of moving from a target to another target is -1 if no edge 
                    exists.
            def_cosntraints (List[Tuple]): list of tuples, each tuple is a pair of resources
                that must be located in close area (within a distance of req_dist).
            atk_eps (float): a small probability that the attacker may choose to attack targets
                with uniform distribution.
            inv_pen (float): penalty for invalid actions.
            req_hop (int): the maximal number of hops required for a defender to move a resource.
            req_dist (int): the maximal distance required for a pair of resources. This is for 
                the defender's assignment constraint.
            edge_matrix_type (str): 'adj' or 'cost', the type of edge matrix as the GNN input.
            att_param (dict): parameters for the attacker's payoff matrix.
                dist_payoff: a vector representing the prob of attacker's payoff plan
                idx_tgt: a matrix where each row represents one payoff plans
        Note:
            This environment has infinite horizon. Please wrap it with `gym.wrappers.TimeLimit`. 
        """
        self.game = SeqSSGBase(num_rsc=num_rsc, num_tgt=num_tgt)

        if has_cost: assert cost_matrix is not None
        self.game._verify_adj_and_cost(adj_matrix, cost_matrix)
        self.process_payoff_matrix(payoff_matrix, att_param)

        self.adj_matrix = adj_matrix
        self.cost_matrix = cost_matrix
        self.def_constraints = np.array(def_constraints)

        self.has_cost = has_cost
        self.has_constraint = has_constraint
        self.num_rsc = num_rsc
        self.num_tgt = num_tgt
        self.init_state = init_state
        self.atk_eps = atk_eps
        self.inv_pen = inv_pen

        self.process_attr_for_testing(attr_for_testing)
        self.fixed_assign = {int(k):int(v) for k,v in fixed_assign.items()} if fixed_assign else None
        self.shortest_path, self.path_cost = self._find_shortest_path_w_path_cost(
            self.adj_matrix, self.cost_matrix
        )
        self.available_move = np.where((self.shortest_path>-1)&(self.shortest_path<=req_hop), 1, 0)
        self.available_asgmt = np.where((self.shortest_path>-1)&(self.shortest_path<=req_dist), 1, 0)

        self.edge_matrix = self._create_edge_matrix(edge_matrix_type)
        
        self.action_space = spaces.MultiDiscrete([self.num_tgt] * self.num_rsc)
        # observation is a concatenation of the features matrix and the adjacency matrix
        #   features matrix: (num_tgt, 2+num_rsc*num_rsc+2), binary values
        #       obs_def_st: (num_tgt, 2) defender's status, indicates if the target is defended, no resource (0), has resource(s) (1)
        #       obs_def_rsc: (num_tgt, num_rsc*num_rsc) indicates the resource assignment, one-hot encoding.
        #       obs_atk: (num_tgt, 2) attack, indicates if the target is attacked, not attacked (0), attacked (1)
        #   adjacency matrix: (num_tgt, num_tgt), min=0, max=1 (can also be cost matrix, a weighed adjacency matrix)
        self.observation_space = spaces.Box(
            shape=(self.num_tgt, self.num_rsc+2+self.num_tgt),
            low=-1, high=self.num_rsc)

        self.tgt_fea = {}
        self.rsc_fea = {}
        self.numba_rsc_fea = typed.Dict.empty(types.int64,types.int64)
        self.numba_cur_rsc_fea = typed.Dict.empty(types.int64,types.int64)

    def reset(self):
        self.tgt_fea = {i: {'def_st': 0, 'def_rsc': [], 'atk': 0} for i in range(self.num_tgt)}
        self.rsc_fea = {i:0 for i in range(self.num_rsc)}
        for k, _ in self.rsc_fea.items(): #copy to numba dict
            self.numba_rsc_fea[k] = 0
        
        if self.init_state is None:
            asg_rsc = np.random.randint(0, self.num_tgt, self.num_rsc) # random assignment of resources
        else:
            asg_rsc = self.init_state.copy() # fixed assignment of resources

        self.upd_by_asgmt(asg_rsc)
        return self.encode_tgt_fea_int(self.tgt_fea)

    def step(self, def_act: np.ndarray):
        
        assert def_act.ndim == 1
        assert def_act.dtype in [np.int32, np.int64]

        # Sample attacker's action
        atk_act = self.sample_atk_action_suqr(self.atk_eps)
        
        # Action validity
        valid = True
        if self.has_constraint:
            validity_cstrs = []
            validity_cstrs.append(self.game.check_adj(def_act, self.available_move, self.numba_rsc_fea))
            validity_cstrs.append(self.game.check_asgmt_cstr(def_act, self.available_asgmt, self.def_constraints))
            validity_cstrs.append(self._check_fixed_assign(def_act))
            valid = np.all(validity_cstrs).item()

        # Perform: move resources
        info = {}
        if not valid:
            self.tgt_fea, def_util, protected, def_util_ori = self.upd_fea_and_pen(atk_act)
            info['rew_ori'] = def_util_ori
        else:
            self.tgt_fea, def_util, protected = self.upd_fea_and_rew(def_act, atk_act)
            info['rew_ori'] = def_util
        next_obs = self.encode_tgt_fea_int(self.tgt_fea)
        info.update({'protected': protected, 'valid': valid})
        return next_obs, def_util, False, info

    #1: not numba friendly- dict not homogenous
    def upd_by_asgmt(self, asgmt: np.ndarray)-> Tuple[dict, dict]:
        """Update defended status and resource location based on the assignment.
        
        Note: 
            Assignment does not change the attack status.
        """
        assert asgmt.ndim == 1
        assert asgmt.dtype in [np.int32, np.int64]

        # reset defended status and resource location, keep attack status unchanged
        self.tgt_fea = {k: {'def_st': 0, 'def_rsc': [], 'atk': v['atk']} 
                        for k, v in self.tgt_fea.items()}
        self.rsc_fea = {k: 0 for k,v in self.rsc_fea.items()}
        for k, _ in self.rsc_fea.items(): #copy to numba dict
            self.numba_rsc_fea[k] = 0

        # perform assignment
        for i, tgt in enumerate(asgmt):
            self.tgt_fea[tgt]['def_st'] = 1
            self.tgt_fea[tgt]['def_rsc'].append(i)
            self.rsc_fea[i] = tgt
            self.numba_rsc_fea[i] = tgt

        return self.tgt_fea, self.rsc_fea

    #2: not numba friendly-  dict not homogenous
    def upd_by_atk(self, atk_act: int):
        """Update attacker's status based on the attack action.
        
        Args:
            atk_act: the target (index) to be attacked.

        Note:
            Attack action does not change the defended status and resource 
            location.
        """
        assert any([isinstance(atk_act, cls) for cls in [int, np.int32, np.int64]])

        # reset attack status, keep defended status and resource location unchanged
        self.tgt_fea = {k: {'def_st': v['def_st'], 'def_rsc': v['def_rsc'], 'atk': 0}
                        for k, v in self.tgt_fea.items()}

        # perform attack
        self.tgt_fea[atk_act]['atk'] = 1

        return self.tgt_fea
    
    #3: not numba friendly- dict not homogenous. 
    def encode_tgt_fea_int(self, tgt_fea) -> np.ndarray:
        """
        Encode defended status, resource location and attack status into a mixed 
            integer matrix, then concatenating with the adjacency matrix.

        Args:
            tgt_fea: a dictionary of target features, with keys as target indices
                and values as dictionaries of features.
        
        Returns:
            An mixed integer matrix of shape (num_tgt, 1+num_rsc+1+num_tgt), where
                the adjacency matrix may be a cost matrix with float values.
        """

        def_st = []
        def_rsc = []
        atk = []
        for i in range(self.num_tgt):
            def_st.append(tgt_fea[i]['def_st']) # List[int]
            def_rsc.append(tgt_fea[i]['def_rsc']) # List[List]
            atk.append(tgt_fea[i]['atk']) # List[int]

        def_st = np.array(def_st).reshape((self.num_tgt, 1)) # (num_tgt, 1)
        def_rsc = np.array([
            np.pad(rsc, (0, self.num_rsc-len(rsc)), 'constant', constant_values=self.num_rsc)
            for rsc in def_rsc]) # Pad the vector for each target to the same length
                                 #      with constant values: num_rsc.
        atk = np.array(atk).reshape((self.num_tgt, 1)) # (num_tgt, 1)
        return np.concatenate((def_st, def_rsc, atk, self.edge_matrix),
                               axis=-1, dtype=np.float32)
    

    #5: not numba friendly- heterogeneous dict used
    def encode_tgt_fea_bianry(self, tgt_fea) -> np.ndarray:
        """
        Encode defended status, resource location and attack status into a binary matrix,
            then concatenating with the adjacency matrix.
        
        Args:
            tgt_fea: a dictionary of target features, with keys as target indices
                and values as dictionaries of features.

        Returns:
            A binary matrix of shape (num_tgt, 2+num_rsc*num_rsc+2+num_tgt).
        """
        def_st = []
        def_rsc = []
        atk = []
        for i in range(self.num_tgt):
            def_st.append(tgt_fea[i]['def_st']) # List[int]
            def_rsc.append(tgt_fea[i]['def_rsc']) # List[List]
            atk.append(tgt_fea[i]['atk']) # List[int]

        def_st = one_hot(np.array(def_st, dtype=np.int32), 2) # (num_tgt, 2)
        def_rsc = np.array([one_hot_w_padding(np.array(rsc, dtype=np.int32), self.num_rsc) 
                            for rsc in def_rsc])
        atk = one_hot(np.array(atk, dtype=np.int32), 2) # (num_tgt, 2)
        return np.concatenate((def_st, def_rsc, atk, self.edge_matrix),
                               axis=-1, dtype=np.float32)

    #6: not numba friendly- heterogeneous dict used
    def sample_atk_action_suqr(self, eps=0.05) -> np.int64:
        """
        Sample attacker's action based on the SUQR algorithm. With a probability of
            epsilon, the attacker will attack a random target.
        """
        atk = np.array([self.tgt_fea[i]['atk'] for i in range(self.num_tgt)])
        not_attacked = np.where(atk < 1)[0] # Indices of targets that are not attacked

        # Epsilon random action
        if np.random.rand() < eps:
            return np.random.choice(not_attacked)

        # Follow the SUQR algorithm
        available_pay = self.payoff_matrix[:, 2:][not_attacked]
        atk_qr = softmax(available_pay.mean(axis=-1))
        atk_action = np.random.choice(not_attacked, p=atk_qr)
        return atk_action

    #7: not numba friendly- heterogeneous dict used
    def upd_fea_and_rew(
        self, 
        def_act: np.ndarray, 
        atk_act: int
    )-> Tuple[np.ndarray, float]:
        """
        Take one step of the game with valid actions.

        Args:
            def_act (np.ndarray): (num_rsc,) array, resource assignment.
            atk_act (int): index of the target to be attacked.
        
        Returns:
            next_tgt_fea (dict): target features of the next state.
            def_util (float): utility of the defender.
        """
        old_rsc_fea = copy.deepcopy(self.rsc_fea) 
        
        # Update status
        self.upd_by_asgmt(def_act)
        self.upd_by_atk(atk_act)

        # Utility
        ## Defending utility
        protected = True if self.tgt_fea[atk_act]['def_st'] == 1 else False
        def_util = self.payoff_matrix[atk_act, 0] if protected else self.payoff_matrix[atk_act, 1]
        
        ## Moving cost
        if self.has_cost:
            pair_locs = [(v, self.rsc_fea[k]) for k,v in old_rsc_fea.items()]
            assert not any([self.path_cost[p_l]<0 for p_l in pair_locs]), \
                "Negative cost detected for a movement pair (t, t')."
            def_util -= sum([self.path_cost[p_l] for p_l in pair_locs])
        return self.tgt_fea, def_util.item(), protected

    #8: not numba friendly- heterogeneous dict used
    def upd_fea_and_pen(
        self,
        atk_act: int,
        chance_to_protect: float=0.5,
    )-> Tuple[dict, float, bool, float]:
        """
        Take one step of the game with invalid actions.

        Args:
            atk_act (int): index of the target to be attacked.
        
        Returns:
            next_tgt_fea (dict): target features of the next state.
            def_pen (float): penalty of the defender for invalid actions.
            protested (bool): whether the target is protected.
            def_util_orig (float): original utility of the defender
                (move no resources).
        """
        # Update status
        self.upd_by_atk(atk_act)

        # Check adjacent constraint (specificed by assignment constraint)
        cur_rsc = np.array([self.rsc_fea[i] for i in range(self.num_rsc)])
        rsc_in_place = self.game.check_asgmt_cstr(cur_rsc, self.available_asgmt, self.def_constraints)

        # Proection status
        # no resouce in place
        if self.tgt_fea[atk_act]['def_st'] != 1:
            protected = False
        # resource in place and supportive resource in place 
        elif rsc_in_place: 
            protected = True
        # resource in place but no supportive resource in place
        else:
            # protect with a chance
            protected = True if np.random.rand() < chance_to_protect else False

        # Utility
        def_util = self.payoff_matrix[atk_act, 0] if protected else self.payoff_matrix[atk_act, 1]

        if self.inv_pen is None: # No penalty, return original utility
            return self.tgt_fea, def_util.item(), protected, def_util.item()
        else: # Return penalty
            raise RuntimeError("No version support invalid action penalty yet as "+
                               "penalty may have negative effects of defender's long-term utilities.")
            return self.tgt_fea, self.inv_pen, protected, def_util.item()


    #11   
    def val_inv_act_gen(self, obs: np.ndarray):
        """
        Generate valid/invalid actions.
            For 1 observation, output 5x valid actions and 5x invalid actions.

        Returns:
            acts: generated actions
            validity: validity of the actions
            obs_ext: the observation extended to the same length as acts
        """
        assert obs.ndim == 3
        threshold = 400

        acts, validity, obs_ext = [], [], []
        for o in obs:
            a, v, o_e = self._val_inv_act_1_obs(o)
            acts.append(a)
            validity.append(v)
            obs_ext.append(o_e)

        acts = np.concatenate(acts)
        validity = np.concatenate(validity)
        obs_ext = np.concatenate(obs_ext)

        if abs((1-validity).sum()- validity.sum()) > threshold:
            warnings.warn("The number of valid actions is much different from\
                 the number of invalid actions.", Warning)
        
        return acts, validity, obs_ext


    #12 wrapper function      
    def act_check(self, def_act: np.ndarray, obs: np.ndarray) -> bool:
        """
        Check if the action is valid given an observation (target feature encoding).
        """

        assert def_act.ndim == 1
        assert def_act.dtype in [np.int32, np.int64]

        # return self.game.fn_act_check(def_act, obs, self.has_constraint, self.available_move, self.numba_rsc_fea)

        if not self.has_constraint: return True

        # Decode the observation
        t, rsc_fea_d = self.decode_tgt_fea_int(obs)
        for k, v in rsc_fea_d.items():
            self.numba_cur_rsc_fea[k] = v

        # Check if the fixed assignment constraint is satisfied
        if not self._check_fixed_assign(def_act):
            return False

        # Check if the assignment satisfies the adjacency constraints
        #     Error if using keyword arguments to check_adj
        if not self.game.check_adj(def_act, self.available_move, self.numba_rsc_fea, self.numba_cur_rsc_fea):
            return False

        # Check if the assignment satisfies the resource constraints
        if not self.game.check_asgmt_cstr(def_act, self.available_asgmt, self.def_constraints):
            return False

        return True


    #13 
    def gen_mask_from_obs(self, obs: np.ndarray, cand_act: np.ndarray) -> np.ndarray:
        """
        Generate mask from observation and candidate actions.
        
        Returns:
            masks: (num_act,) array, 1 if the action is valid, 0 otherwise.
        """
        assert obs.ndim == 2
        assert cand_act.ndim == 2
        return np.apply_along_axis(self.act_check, 1, cand_act, obs)


    #15: not numba friendly- networkx module not numba friendly
    def _find_shortest_path_w_path_cost(
        self, 
        adj: np.ndarray,
        cost: np.ndarray=None, 
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Find the shortest paths of all pairs in a graph.
        
        Args:
            cost (np.ndarray): (num_tgt, num_tgt) array, cost matrix. 
            adj (np.ndarray): (num_tgt, num_tgt) array, adjacency matrix.

        Returns: 
            shortest_path (np.ndarray): (num_tgt, num_tgt) array, shortest 
                path length matrix. No path is represented by -1.
            path_cost (np.ndarray | None): (num_tgt, num_tgt) array, shortest 
                path cost matrix. No path is represented by -1. If cost is None,
                path_cost is None.
        """
        G = nx.from_numpy_matrix(adj, parallel_edges=False)
        p = nx.shortest_path(G)

        shortest_path = np.zeros_like(adj, dtype=np.int32)-1
        path_cost = (np.zeros_like(adj, dtype=np.float32)-1.0) \
            if cost is not None else None
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                try:
                    path = p[i][j]
                    shortest_path[i, j] = len(path) - 1
                    if cost is not None: 
                        path_cost[i, j] = sum([cost[path[k], path[k+1]] 
                                              for k in range(len(path)-1)])
                except KeyError:
                    pass
        return shortest_path, path_cost

    #16: not numba friendly- no numba support for deepcopy
    def _create_edge_matrix(self, mat_type: str) -> np.ndarray:
        """Create edge matrix."""
        if mat_type == 'adj':
            return copy.deepcopy(self.adj_matrix)
        elif mat_type == 'cost':
            return copy.deepcopy(self.cost_matrix)

    #17       
    def _val_inv_act_1_obs(self, obs: np.ndarray, quota: int=1000):
        """Generate valid/invalid actions for 1 observation."""
        batch = 20
        n_val = n_inv = 5


        # Decode the observation
        tgt_fea, rsc_fea = self.decode_tgt_fea_int(obs)

        # Generate valid/invalid actions
        val_act, inv_act = [], []
        opts = [None for _ in range(self.num_rsc)]
        ## Valid actions
        for r, t in rsc_fea.items():
            a_m = self.available_move[t]
            opts[r] = np.where(a_m)[0]
        for _ in range(quota//batch):
            act = np.array([sample_cartesian_product(opts) for _ in range(batch)])
            val = np.apply_along_axis(self.act_check, 1, act, obs)
            if len(val_act)+val.sum() >= n_val:
                val_act += act[val][:(n_val-len(val_act))].tolist()
                break
            else:
                val_act += val_act + act[val].tolist()
        
        ## Invalid actions
        for _ in range(quota//batch):
            act = np.random.randint(0, self.num_tgt, size=(batch, self.num_rsc))
            val = np.apply_along_axis(self.act_check, 1, act, obs)
            inv = (~val)
            if len(inv_act)+inv.sum() >= n_inv:
                inv_act += act[inv][:(n_inv-len(inv_act))].tolist()
                break
            else:
                inv_act += act[inv].tolist()

        acts = np.array(val_act+inv_act)
        validity = np.concatenate([np.ones(len(val_act), dtype=np.int32), 
                                   np.zeros(len(inv_act), dtype=np.int32)])
        obs_ext = repeat(obs, '... -> b ...', b=len(acts))
        
        return acts, validity, obs_ext

    #4: not numba friendly- heterogeneous dictionary is used
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
        
        # print("decode tgt fea", tgt_fea)
        # print("decode rsc fea", tgt_fea)
        return tgt_fea, rsc_fea

    def process_payoff_matrix(self, payoff_matrix: np.ndarray, att_param: dict=None):
        """
        Process payoff matrix.
            Change the payoff of the attacker 
        
        """
        # Only one payoff plan
        if att_param is None:
            self._payoff_matrix = payoff_matrix
            return
        
        # Multiple payoff plans
        self._payoff_matrix = []
        dist_payoff = att_param['dist_payoff']
        idx_tgt = att_param['idx_tgt']
        for idx in idx_tgt:
            payoff = np.concatenate([payoff_matrix[:, :2], payoff_matrix[:, 2:][idx]], axis=1)
            self._payoff_matrix.append(payoff)
        self.dist_payoff = dist_payoff
    
    def process_attr_for_testing(self, attr_for_testing: dict=None):
        if attr_for_testing is None:
            self.attr_for_testing = None
        else:
            for attr_name in attr_for_testing:
                # assume each attribute is a dictionary and the keys should be integers
                attr_for_testing[attr_name] = {int(k): v for k, v in attr_for_testing[attr_name].items()}
            self.attr_for_testing = attr_for_testing

    def get_attr_for_testing(self):
        return self.attr_for_testing

    @property
    def payoff_matrix(self):
        """Return payoff matrix."""
        if isinstance(self._payoff_matrix, np.ndarray):
            return self._payoff_matrix
        elif isinstance(self._payoff_matrix, list):
            idx_plan = np.random.choice(len(self._payoff_matrix), p=self.dist_payoff)
            return self._payoff_matrix[idx_plan]
        else:
            raise RuntimeError("Payoff matrix is not initialized.")
        
    def _check_fixed_assign(self, asgmt: np.ndarray):
        """Check if the assignment is fixed.
        
        Note:
            fixed_assign (dict): key is the resource index, value is the target index.
        """
        if self.fixed_assign is None:
            return True
        else:
            return np.all([t == asgmt[r] for r, t in self.fixed_assign.items()])

# ================== Legacy Code ==================

    def asgmt2def_obs(self, asgmt: np.ndarray):
        """
        Convert resource assignment to defender observation.

        Args:
            asgmt (np.ndarray): (num_rsc, ) resource assignment, each element is the index of the target.

        Returns:
            obs_def_st (np.ndarray): (num_tgt, 2) defender's status, indicates if the target is defended, 
                no resource (0), has resource(s) (1).
            obs_def_rsc (np.ndarray): (num_tgt, num_rsc*num_rsc) indicates the resource assignment, one-hot 
                encoding.
        """
        raise RuntimeError("Function is not compatible with new implementation.")
        assert asgmt.ndim == 1
        assert asgmt.dtype in [np.int32, np.int64]

        obs_def = [{'def': 0, 'rsc': []} for _ in range(self.num_tgt)]
        for i, t in enumerate(asgmt):
            obs_def[t]['rsc'].append(i)
            obs_def[t]['def'] = 1
        obs_def_st = one_hot(np.array([d['def'] for d in obs_def], dtype=np.int32), 2)
        obs_def_rsc = np.array([one_hot_w_padding(
            np.array(x['rsc'], dtype=np.int32), self.num_rsc) for x in obs_def])
        return obs_def_st, obs_def_rsc
    
    def def_obs2asgmt(self, obs_def_rsc: np.ndarray):
        """
        Convert defender observation to resource assignment.

        Args:
            obs_def_rsc (np.ndarray): (num_tgt, num_rsc*num_rsc) indicates the resource assignment, one-hot 
                encoding.

        Returns:
            asgmt (np.ndarray): (num_rsc, ) resource assignment, each element is the index of the target.
        """
        raise RuntimeError("Function is not compatible with new implementation.")
        assert obs_def_rsc.ndim == 2

        safe_argmax = lambda x: np.argmax(x, axis=-1) if any(x) else -1
        obs_def_rsc = rearrange(obs_def_rsc, 't (r1 r2) -> t r1 r2', r1=self.num_rsc)
        obs_def_rsc = np.apply_along_axis(safe_argmax, axis=-1, arr=obs_def_rsc)
        asgmt = np.zeros((self.num_rsc,), dtype=np.int32)
        for i, rsc in enumerate(obs_def_rsc):
            rsc = [x for x in rsc if x != -1]
            for j in rsc:
                asgmt[j] = i
        
        return asgmt
    
    def uni2legacy(self, state: np.ndarray, dtype=np.int32):
        """Convert from universal state to legacy state.

        Args:
            state (np.ndarray): universal state, shape (num_tgt, num_rsc+2).
                (:, :num_rsc) is the defender's action (assignment of resources)
                (:, num_rsc:) is the attacker's action

        Returns:
            np.ndarray: legacy state, shape (num_tgt, 2).
                (:, 0) is the defender's action
                (:, 1) is the attacker's action
        """
        raise RuntimeError("Function is not compatible with new implementation.")

        state_local = state.copy()
        state_legecy = np.concatenate(
            (np.argmax(state_local[:, :self.num_rsc], axis=1, keepdims=True),
             np.argmax(state_local[:, self.num_rsc:], axis=1, keepdims=True)),
            axis=-1, dtype=dtype
        )
        return state_legecy
        
    def legacy2uni(self, state: np.ndarray, dtype=np.int32):
        """Convert from legacy state to universal state.

        Args:
            state (np.ndarray): legacy state, shape (num_tgt, 2).
                (:, 0) is the defender's action
                (:, 1) is the attacker's action

        Returns:
            np.ndarray: universal state, shape (num_tgt, num_rsc+2).
                (:, :num_rsc) is the defender's action (assignment of resources)
                (:, num_rsc:) is the attacker's action
        """
        raise RuntimeError("Function is not compatible with new implementation.")

        state_local = state.copy()
        state_uni = np.concatenate(
            (one_hot(state_local[:, 0], self.num_rsc),
            one_hot(state_local[:, 1], 2)),
            axis=-1, dtype=dtype
        )
        return  state_uni