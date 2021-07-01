from blackjack_rl.agent.agent import Agent
from blackjack_rl.utils.typedef import State, Action, Reward, Trans
import numpy as np
from collections import defaultdict
from typing import Tuple, List


class QTableAgent(Agent):
    def __init__(self):
        # vector w
        self.qtable = defaultdict(lambda: np.zeros(2))
        self.gamma = 0.9
        self.alpha = 0.02
        self.coef = 0.0001
        self.train_count = 0

    # select action
    def take_action(self, state: State) -> Action:
        # greedy action
        if state[1] < 12:
            return True
        if (state not in self.qtable) or (np.random.uniform(0, 1) < 1.0/(self.train_count * self.coef + 1)):
            return np.random.choice([True, False])
        else:
            return self.qtable[state][1] > self.qtable[state][0]

    # update weight from train_data
    # return whether weight is updated or not
    def train(self, train_data: List[Trans], monte: bool = True) -> bool:
        # only use appropriate data
        train_data = [(d[0], d[1], d[2], d[3]) for d in train_data if self._isvalid(d[0])]
        if not train_data:
            return False
        # A = self._calculate_A(train_data)
        # b = self._calculate_b(train_data)
        G = train_data[-1][2]
        different = False
        train_data.reverse()
        for trans in train_data:
            st, at, rtt, stt = trans
            at = int(at)
            old_val = self.qtable[st][at]
            if monte:
                self.qtable[st][at] += self.alpha * (G - self.qtable[st][at])
                G *= self.gamma
            else:
                q_next = 0
                if (not at) and (self._isvalid(stt)):
                    q_next = self.qtable[stt][at]
                self.qtable[st][at] += self.alpha * (rtt + q_next - self.qtable[st][at])
            different = different or (self.qtable[st][at] != old_val)
        self.train_count += 1
        return different

    # included in state space
    @staticmethod
    def _isvalid(state: State) -> bool:
        return (2 <= state[0] <= 11) and (12 <= state[1] <= 20)

