from blackjack_rl.agent import Agent
from typing import Tuple, List, Union
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv

# index = 0
dealer_offset = 2
player_offset = 12
# state space(3dims) × action space(1dims)
all_space = (10, 9, 2, 2)
# all space size
all_zize = np.prod(all_space)
# regularize factor
episilon = 0.01


class LSPIAgent(Agent):
    def __init__(self):
        # vector w
        self.weight = np.zeros(all_zize, dtype=float)

    def reset_weight(self):
        self.weight = np.zeros(all_zize, dtype=float)

    def take_action(self, state: Tuple[int, int, bool]) -> bool:
        # greedy action
        if state[1] < 12:
            return True
        # compare action-values
        # if equal, take action "True"
        return self.weight[self._translate_weight_idx(state=self._reindex_state(state), action=True)] \
               >= self.weight[self._translate_weight_idx(state=self._reindex_state(state), action=False)]

    def train(self, data: List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]):
        # only use appropriate data
        data = [(self._reindex_state(d[0]), d[1], d[2], d[3]) for d in data if self._isvalid(d[0])]
        if not data: return
        A = self._calculate_A(data)
        b = self._calculate_b(data)
        # after regularization, inverse matrix
        new_weight = inv(A + csc_matrix(episilon*np.eye(all_zize, dtype=float), dtype=float)) * b
        self.weight = new_weight.toarray().ravel()

    def _calculate_A(self, data: List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]) -> csc_matrix:
        A = csc_matrix((all_zize, all_zize), dtype=float)
        for d in data:
            st, at, rtt, stt = d
            phi_t, phi_tt = csc_matrix((all_zize, 1), dtype=float), csc_matrix((all_zize, 1), dtype=float)
            phi_t[self._translate_weight_idx(st, at), 0] = 1
            if self._isvalid(state=stt):
                stt = self._reindex_state(stt)
                phi_tt[self._translate_weight_idx(stt, self.take_action(state=stt)), 0] = 1
            td_phi = phi_t - phi_tt
            td_phi = csc_matrix(td_phi.toarray().T)
            A += phi_t * td_phi
        return A

    def _calculate_b(self, data: List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]) -> csc_matrix:
        b = csc_matrix((all_zize, 1), dtype=float)
        for d in data:
            st, at, rtt, stt = d
            phi_t = csc_matrix((all_zize, 1), dtype=float)
            phi_t[self._translate_weight_idx(st, at)] = 1
            b += phi_t * rtt
        return b

    # included in state space
    def _isvalid(self, state: Tuple[int, int, bool]) -> bool:
        return (2 <= state[0] <= 11) and (12 <= state[1] <= 20)

    # reindex dealer and player state
    def _reindex_state(self, state: Tuple[int, int, bool]) -> Tuple[int, int, bool]:
        assert self._isvalid(state)
        dealer, player, have = state
        return dealer - dealer_offset, player - player_offset, have

    # function phi(s,a): (s,a) -> R^(all_size)
    def _translate_weight_idx(self, state: Tuple[int, int, bool], action: bool) -> int:
        offset = 1
        idx = 0
        all = state + (action,)
        for i, val in enumerate(all):
            assert 0 <= val < all_space[i]
            idx += int(val) * offset
            offset *= all_space[i]
        return idx
