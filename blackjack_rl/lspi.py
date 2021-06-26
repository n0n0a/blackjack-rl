from blackjack_rl.agent import Agent
from typing import Tuple, List, Union
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from blackjack_rl.typedef import State, Action, Trans

# index = 0
dealer_offset = 2
player_offset = 12
# state space(3dims) × action space(1dims)
all_space = (10, 9, 2, 2)
# all space size
all_size = np.prod(all_space)
# regularize factor
episilon = 0.000001


class LSPIAgent(Agent):
    def __init__(self):
        # vector w
        self.weight = np.zeros(all_size, dtype=float)
        # for dsum in range(all_space[0]):
        #     for psum in range(all_space[1]):
        #         for ace in [True, False]:
        #             if psum < 20:
        #                 self.weight[self._translate_weight_idx((dsum, psum, ace), True)] = 1.0
        #                 self.weight[self._translate_weight_idx((dsum, psum, ace), False)] = -1.0
        #             else:
        #                 self.weight[self._translate_weight_idx((dsum, psum, ace), True)] = -1.0
        #                 self.weight[self._translate_weight_idx((dsum, psum, ace), False)] = 1.0

    # set weight = 0
    def reset_weight(self):
        self.weight = np.zeros(all_size, dtype=float)

    # select action
    def take_action(self, state: State) -> Action:
        # greedy action
        if state[1] < 12:
            return True
        # compare action-values
        # if equal, take action "False"
        return self.weight[self._translate_weight_idx(state=self._reindex_state(state), action=True)] \
               > self.weight[self._translate_weight_idx(state=self._reindex_state(state), action=False)]

    def train(self, train_data: List[Trans], epochs: int = 1):
        for _ in range(epochs):
            # only use appropriate data
            train_data = [(self._reindex_state(d[0]), d[1], d[2], d[3]) for d in train_data if self._isvalid(d[0])]
            if not train_data: return
            A = self._calculate_A(train_data)
            b = self._calculate_b(train_data)
            # after regularization, inverse matrix
            # new_weight = inv(A) * b
            new_weight = inv(A + csc_matrix(episilon * np.eye(all_size, dtype=float), dtype=float)) * b
            self.weight = new_weight.toarray().ravel()

    # calculate A for updating weight
    def _calculate_A(self, train_data: List[Trans]) -> csc_matrix:
        A = csc_matrix((all_size, all_size), dtype=float)
        for d in train_data:
            st, at, rtt, stt = d
            phi_t, phi_tt = csc_matrix((all_size, 1), dtype=float), csc_matrix((all_size, 1), dtype=float)
            phi_t[self._translate_weight_idx(st, at), 0] = 1
            if self._isvalid(state=stt):
                phi_tt[self._translate_weight_idx(self._reindex_state(stt), self.take_action(state=stt)), 0] = 1
            td_phi = phi_t - phi_tt
            td_phi = csc_matrix(td_phi.toarray().T)
            A += phi_t * td_phi
        return A

    # calculate b for updating weight
    def _calculate_b(self, train_data: List[Trans]) -> csc_matrix:
        b = csc_matrix((all_size, 1), dtype=float)
        for d in train_data:
            st, at, rtt, stt = d
            phi_t = csc_matrix((all_size, 1), dtype=float)
            phi_t[self._translate_weight_idx(st, at)] = 1
            b += phi_t * rtt
        return b

    # included in state space
    def _isvalid(self, state: State) -> bool:
        return (2 <= state[0] <= 11) and (12 <= state[1] <= 20)

    # reindex dealer and player state
    def _reindex_state(self, state: State) -> State:
        assert self._isvalid(state)
        dealer, player, have = state
        return dealer - dealer_offset, player - player_offset, have

    # function phi(s,a): (s,a) -> R^(all_size)
    def _translate_weight_idx(self, state: State, action: Action) -> int:
        offset = 1
        idx = 0
        all = state + (action,)
        for i, val in enumerate(all):
            assert 0 <= val < all_space[i]
            idx += int(val) * offset
            offset *= all_space[i]
        return idx
