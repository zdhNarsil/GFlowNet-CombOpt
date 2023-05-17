import os, sys
from itertools import count
import random
import pathlib
import ipdb
import functools

import math
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import dgl
import dgl.function as fn


######### Pytorch Utils

def seed_torch(seed, verbose=True):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    if verbose:
        print("==> Set seed to {:}".format(seed))

def pad_batch(vec, dim_per_instance, padding_value, dim=0, batch_first=True):
    # dim_per_instance: list of int
    tupllle = torch.split(vec, dim_per_instance, dim=dim)
    pad_tensor = pad_sequence(tupllle, batch_first=batch_first, padding_value=padding_value)
    return pad_tensor

def ema_update(model, ema_model, alpha=0.999):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

######### MDP Utils

def get_decided(state, task="MaxIndependentSet") -> torch.bool:
    # assert isinstance(state, torch.LongTensor) or isinstance(state, torch.cuda.LongTensor) # cannot be used in jit
    assert state.dtype == torch.long
    if task in ["MaxIndependentSet", "MinDominateSet", "MaxClique", "MaxCut"]:
        return state != 2
    else:
        raise NotImplementedError

# in unif pb, to calculate number of parents
# also the number of steps taken (i.e. reward)
def get_parent(state, task="MaxIndependentSet") -> torch.bool:
    assert state.dtype == torch.long
    if task in ["MaxIndependentSet", "MaxClique", "MaxCut"]:
        return state == 1
    elif task in ["MinDominateSet"]:
        return state == 0
    else:
        raise NotImplementedError


class GraphCombOptMDP(object):
    def __init__(self, gbatch, cfg):
        self.cfg = cfg
        self.task = cfg.task
        self.device = gbatch.device
        self.gbatch = gbatch
        self.batch_size = gbatch.batch_size
        self.numnode_per_graph = gbatch.batch_num_nodes().tolist()
        cum_num_node = gbatch.batch_num_nodes().cumsum(dim=0)
        self.cum_num_node = torch.cat([torch.tensor([0]).to(cum_num_node), cum_num_node])[:-1]
        self._state = torch.full((gbatch.num_nodes(),), 2, dtype=torch.long, device=self.device)
        self.done = torch.full((self.batch_size,), False, dtype=torch.bool, device=self.device)

    @property
    def state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def get_decided_mask(self, state=None):
        state = self._state if state is None else state
        return get_decided(state, self.task)

    def step(self, action):
        raise NotImplementedError

    def get_log_reward(self):
        raise NotImplementedError

    def batch_metric(self, state): # return a list of metric
        raise NotImplementedError

def get_mdp_class(task):
    if task == "MaxIndependentSet":
        return MaxIndSetMDP
    elif task == "MaxClique":
        return MaxCliqueMDP
    elif task == "MinDominateSet":
        return MinDominateSetMDP
    elif task == "MaxCut":
        return MaxCutMDP
    else:
        raise NotImplementedError

class MaxIndSetMDP(GraphCombOptMDP):
    # MDP conditioned on a batch of graphs
    # state: 0: not selected, 1: selected, 2: undecided
    def __init__(self, gbatch, cfg):
        assert cfg.task == "MaxIndependentSet"
        super(MaxIndSetMDP, self).__init__(gbatch, cfg)

    # @profile
    def step(self, action):
        state = self._state.clone()

        # label the selected node to be "1"
        action_node_idx = (self.cum_num_node + action)[~self.done]
        # make sure the idx of action hasn't been decided
        assert torch.all(~self.get_decided_mask(state[action_node_idx]))
        state[action_node_idx] = 1

        # label all nodes near the selected node ("1") to be "0"
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = (state == 1).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x1_deg = self.gbatch.ndata.pop('h')  # (#node, # of 1-labeled neighbour node)
        undecided = ~get_decided(state)
        state[undecided & (x1_deg > 0)] = 0
        self._state = state

        decided_tensor = pad_batch(self.get_decided_mask(state), self.numnode_per_graph, padding_value=True)
        self.done = torch.all(decided_tensor, dim=1)
        return state

    def get_log_reward(self):
        state = pad_batch(self._state, self.numnode_per_graph, padding_value=2)
        sol = (state == 1).sum(dim=1).float()
        return sol

    def batch_metric(self, vec_state):
        state_per_graph = torch.split(vec_state, self.numnode_per_graph, dim=0)
        return [(s == 1).sum().item() for s in state_per_graph]

class MaxCliqueMDP(GraphCombOptMDP):
    # initial state: all nodes = "2" (all nodes are undecided)
    # 1: selected, 0: not selected, 2: undecided
    def __init__(self, gbatch, cfg,):
        super(MaxCliqueMDP, self).__init__(gbatch, cfg)

    def step(self, action):
        state = self._state.clone()

        # label the selected node to be "1"
        action_node_idx = (self.cum_num_node + action)[~self.done]
        assert torch.all(~self.get_decided_mask(state[action_node_idx]))
        state[action_node_idx] = 1

        # calculate num of "1" for each grpah
        num1 = pad_batch(state == 1, self.numnode_per_graph, padding_value=0).sum(dim=1)
        num1 = [num * torch.ones(count).to(self.device) for count, num in zip(self.numnode_per_graph, num1)]
        num1 = torch.cat(num1) # same shape with state
        # if a node is not connected to all "1" nodes, label it to be "0"
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = (state == 1).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x1_deg = self.gbatch.ndata.pop('h')
        undecided = ~get_decided(state)
        state[undecided & (x1_deg < num1)] = 0
        self._state = state

        decided_tensor = pad_batch(self.get_decided_mask(state), self.numnode_per_graph, padding_value=True)
        self.done = torch.all(decided_tensor, dim=1)
        return state

    def get_log_reward(self):
        state = pad_batch(self._state, self.numnode_per_graph, padding_value=2)
        sol = (state == 1).sum(dim=1).float()
        return sol

    def batch_metric(self, vec_state):
        state_per_graph = torch.split(vec_state, self.numnode_per_graph, dim=0)
        return [(s == 1).sum().item() for s in state_per_graph]

class MinDominateSetMDP(GraphCombOptMDP):
    # initial state: all nodes = "2" (all nodes are in the set)
    # 0: already deleted from the set, 1: in set, can't be deleted from the set,
    # 2: in set, might be deleted from the set in future steps
    def __init__(self, gbatch, cfg):
        super(MinDominateSetMDP, self).__init__(gbatch, cfg)
        assert not cfg.back_trajectory

    def step(self, action):
        state = self._state.clone()

        # action: delete a node from set (label it to be "0" from "2")
        action_node_idx = (self.cum_num_node + action)[~self.done]
        # assert torch.all(state[action_node_idx] == 2)
        assert torch.all(~self.get_decided_mask(state[action_node_idx]))
        state[action_node_idx] = 0

        undecided = ~get_decided(state)
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = ((state == 1) | (state == 2)).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x12_deg = self.gbatch.ndata.pop('h').int()
        # or if a "2" has no neighbour in the set, it must stay in the set, too
        state[undecided & (x12_deg == 0)] = 1

        # this kinds of special "0" needs to have a neighbour stay in the set
        special0 = (state == 0) & (x12_deg <= 1)
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = special0.float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            xsp0_deg = self.gbatch.ndata.pop('h').int()
        state[undecided & (xsp0_deg >= 1)] = 1

        # for the rest "2" node: it and all its neighbours are connected to the set
        # thus can be deleted from the set
        self._state = state

        decided_tensor = pad_batch(self.get_decided_mask(state), self.numnode_per_graph, padding_value=True)
        self.done = torch.all(decided_tensor, dim=1)
        return state

    def get_log_reward(self):
        state = pad_batch(self._state, self.numnode_per_graph, padding_value=2)
        sol = - (state == 1).sum(dim=1).float() # todo
        # sol = (state == 0).sum(dim=1).float()
        return sol

    def batch_metric(self, vec_state):
        # size of the set: lower is better
        # thus we return negative value for eval convenience
        state_per_graph = torch.split(vec_state, self.numnode_per_graph, dim=0)
        # use negative to be consistent with anneal alg implementation
        return [-(s == 1).sum().item() for s in state_per_graph]

class MaxCutMDP(GraphCombOptMDP):
    # initial state: all nodes = "2" (all nodes are NOT in the set)
    # 0: chosen to not be in the set
    # 1: chosen to be in the set
    # 2: undecided, also not in the set
    def __init__(self, gbatch, cfg):
        super(MaxCutMDP, self).__init__(gbatch, cfg)
        assert not cfg.back_trajectory

    def step(self, action):
        state = self._state.clone()

        # action: choose a node to be in the set (label it to be "1" from "2")
        action_node_idx = (self.cum_num_node + action)[~self.done]
        assert torch.all(~self.get_decided_mask(state[action_node_idx]))
        state[action_node_idx] = 1

        undecided = ~get_decided(state)
        # if a "2" has more "1" neighbours than "0"or"2" neighbours
        # it must NOT be in the set, thus label it to be "0"
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = (state == 1).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x1_deg = self.gbatch.ndata.pop('h').int()
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = ((state == 0) | (state == 2)).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x02_deg = self.gbatch.ndata.pop('h').int()
        state[undecided & (x1_deg > x02_deg)] = 0
        self._state = state

        decided_tensor = pad_batch(self.get_decided_mask(state), self.numnode_per_graph, padding_value=True)
        self.done = torch.all(decided_tensor, dim=1)
        return state

    def get_log_reward(self, state=None): # calculate the cut
        if state is None:
            state = self._state.clone()
        state[state == 2] = 0 # "0" for "not in the set"
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = state.float()
            self.gbatch.apply_edges(fn.u_add_v("h", "h", "e"))
            # 0 + 0 = 0 (not cut), 0 + 1 = 1 (cut), 1 + 1 = 2 (not cut)
            self.gbatch.edata["e"] = (self.gbatch.edata["e"] == 1).float()
            cut = dgl.sum_edges(self.gbatch, 'e') # (bs, )
        cut = cut / 2 # each edge is counted twice
        return cut

    def batch_metric(self, vec_state):
        return self.get_log_reward(vec_state).tolist()


######### Replay Buffer Utils

from multiprocessing import Pool
def imap_unordered_bar(func, args, n_processes=2):
    p = Pool(n_processes)
    args = list(args)
    res_list = []
    for i, res in enumerate(p.imap_unordered(func, args)):
        if isinstance(res, list):
            res_list.extend(res)
        else:
            res_list.append(res)
    p.close()
    p.join()
    return res_list


class TransitionBuffer(object):
    def __init__(self, size, cfg):
        self.size = size
        self.buffer = []
        self.pos = 0

    def reset(self):
        self.buffer = []
        self.pos = 0

    def add_batch(self, batch):
        gb, traj_s, traj_a, traj_d, traj_r, traj_len = batch
        numnode_per_graph = gb.batch_num_nodes().tolist()
        batch_size = gb.batch_size  # num_graph
        g_list = dgl.unbatch(gb)
        traj_s_tuple = torch.split(traj_s, numnode_per_graph, dim=0)

        for b_idx in range(batch_size):
            g_bidx = g_list[b_idx]
            traj_len_bidx = traj_len[b_idx]
            traj_s_bidx = traj_s_tuple[b_idx][..., :traj_len_bidx]
            traj_a_bidx = traj_a[b_idx, :traj_len_bidx - 1]
            traj_d_bidx = traj_d[b_idx, 1:traj_len_bidx] # "done" after transition
            traj_r_bidx = traj_r[b_idx, :traj_len_bidx]

            for i in range(traj_len_bidx - 1):
                transition = (g_bidx, traj_s_bidx[:, i], traj_r_bidx[i], traj_a_bidx[i],
                              traj_s_bidx[:, i + 1], traj_r_bidx[i+1], traj_d_bidx[i])
                self.add_single_transition(transition)

    def add_single_transition(self, inp):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.pos] = inp
        self.pos = (self.pos + 1) % self.size

    @staticmethod
    def transition_collate_fn(transition_ls):
        gbatch, s_batch, logr_batch, a_batch, s_next_batch, logr_next_batch, d_batch = \
            zip(*transition_ls)  # s_batch is a list of tensors
        gbatch = dgl.batch(gbatch)

        s_batch = torch.cat(s_batch, dim=0)  # (sum of # nodes in batch, )
        s_next_batch = torch.cat(s_next_batch, dim=0)

        logr_batch = torch.stack(logr_batch, dim=0)
        logr_next_batch = torch.stack(logr_next_batch, dim=0)
        a_batch = torch.stack(a_batch, dim=0)
        d_batch = torch.stack(d_batch, dim=0)

        return gbatch, s_batch, logr_batch, a_batch, s_next_batch, logr_next_batch, d_batch

    def sample(self, batch_size):
        # random.sample: without replacement
        batch = random.sample(self.buffer, batch_size) # list of transition tuple
        return self.transition_collate_fn(batch)

    def sample_from_indices(self, indices):
        batch = [self.buffer[i] for i in indices]
        return self.transition_collate_fn(batch)

    def __len__(self):
        return len(self.buffer)