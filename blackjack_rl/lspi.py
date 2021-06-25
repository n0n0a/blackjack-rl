from blackjack_rl.agent import Agent
from typing import Tuple, List, Union
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import inv

state_space = (10, 9, 2)
action_space = (2)
all_space = state_space + action_space
all_zize = np.prod(all_space)
episilon = 0.01

class LSPIAgent(Agent):
    def __init__(self):
        self.weight = np.zeros(all_zize, dtype=int)

    @staticmethod
    def translate_index(state: Tuple[int, int, bool], action: bool) -> int:
        offset = 1
        idx = 0
        all = state + tuple(action)
        for i, val in all:
            assert 0 <= val < all_space[i]
            idx += int(val) * offset
            offset *= all_space[i]
        return idx

    def take_action(self, state: Tuple[int, int, bool]) -> bool:
        if state[0] < 12:
            return True
        return self.weight[self.translate_index(state=state, action=True)] \
               >= self.weight[self.translate_index(state=state, action=False)]

    def train(self, data: List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]):
        A = self._calculate_A(data)
        b = self._calculate_b(data)
        self.weight = inv(A + csr_matrix(episilon*np.eye(all_zize, dtype=float))) * b

    def _calculate_A(self, data: List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]) -> csr_matrix:
        A = csr_matrix((all_zize, all_zize), dtype=float)
        for d in data:
            st, at, rtt, stt = d
            phi_t, phi_tt = csr_matrix((all_zize, 1), dtype=float), csr_matrix((all_zize, 1), dtype=float)
            phi_t[self.translate_index(st,at)] = 1
            phi_tt[self.translate_index(stt, self.take_action(state=stt))] = 1
            td_phi = phi_t - phi_tt
            td_phi = csr_matrix(td_phi.toarray().T)
            A += phi_t * td_phi
        return A

    def _calculate_b(self, data: List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]) -> csr_matrix:
        b = csr_matrix((all_zize, all_zize), dtype=float)
        for d in data:
            st, at, rtt, stt = d
            phi_t = csr_matrix((all_zize, 1), dtype=float)
            phi_t[self.translate_index(st,at)] = 1
            b += phi_t * rtt
        return b