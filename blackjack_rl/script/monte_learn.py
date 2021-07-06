from blackjack_rl.envs.eleven_ace import BlackjackEnv
from blackjack_rl.agent.qtable import QTableAgent
import os, pickle
import datetime

# environment seed
seed = 3
# LSPI train count
N_epoch = 100
# make_sample episode count
N_episode = 10000
# Evaluation count per leaning
N_eval = 10000
# data dir
_base = os.path.dirname(os.path.abspath(__file__))#実行中のファイル(このファイル)の絶対パス
data_dir = os.path.join(_base, "../../data")#実行中のファイルからの相対パスでdataの出力先を決定
detail_dir = os.path.join(_base, "../../data/detail")

if __name__ == '__main__':
    # initialize
    env = BlackjackEnv(seed=seed)
    agent = QTableAgent()

    # learning
    rewards = []
    for epoch in range(N_epoch):
        for episode in range(N_episode):
            result = env.run_one_game(agent=agent)
            agent.train(result)
        mean = 0.0
        for _ in range(N_eval):
            result = env.run_one_game(agent=agent)
            if _ == 0:
                print(result)
            mean += result[-1][2]
        mean /= N_eval
        # think reward mean as performance
        rewards.append(mean)
        print(f"epoch:{epoch} performance:{mean}")
        # resampling

    # save result
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)
    now = datetime.datetime.now()
    with open(os.path.join(data_dir, "monte_rewards.pkl"), "wb") as f:
        pickle.dump(rewards, f)
    with open(os.path.join(detail_dir, "monte_rewards_"+now.strftime('%Y%m%d_%H%M%S')+".pkl"), "wb") as f:
        pickle.dump(rewards, f)