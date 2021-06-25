from blackjack_rl.environment import BlackjackEnv
from blackjack_rl.lspi import LSPIAgent
import os

# LSPI train count
N_train = 100
# LSPI test count
N_test = 10
data_dir = "../data"


if __name__ == '__main__':
    # 初期化
    env = BlackjackEnv(seed=0)
    agent = LSPIAgent()

    # 訓練データ作成
    samples = env.make_samples(episode=10000)

    # 学習
    for epoch in N_train:
        agent.train(data=samples)
        # TODO: evaluate LSPI policy
        # TODO: store results

    # TODO: save results
