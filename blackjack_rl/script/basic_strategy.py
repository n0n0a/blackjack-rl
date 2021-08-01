from blackjack_rl.envs.eleven_ace import BlackjackEnv
import os, pickle
import datetime

# environment seed
seed = 3
# make_sample episode count
N_epoch = 10
# LSPI train count
N_episode = 10000
# Evaluation count per leaning
N_eval = 10000
# data dir
_base = os.path.dirname(os.path.abspath(__file__))  # 実行中のファイル(このファイル)の絶対パス
data_dir = os.path.join(_base, "../../data")  # 実行中のファイルからの相対パスでdataの出力先を決定
detail_dir = os.path.join(_base, "../../data/detail")

hard_policy = [[True, True, False, False, False, True, True, True, True, True],
               [False, False, False, False, False, True, True, True, True, True],
               [False, False, False, False, False, True, True, True, True, True],
               [False, False, False, False, False, True, True, True, True, True],
               [False, False, False, False, False, True, True, True, True, True]]

soft_policy = [False, True, True, True, True, False, False, True, True, True]


def basic_strategy(state):
    dealer, player, ace = state
    assert 2 <= dealer <= 11
    if ace:
        if player == 18:
            return soft_policy[dealer - 2]
        elif player < 18:
            return True
        else:
            return False
    else:
        if player < 12:
            return True
        elif player > 16:
            return False
        else:
            return hard_policy[player - 12][dealer - 2]
    assert False


if __name__ == '__main__':
    # initialize
    env = BlackjackEnv(seed=seed)

    # rewards = []
    # weights_eleven = []
    # weights_one = []
    # weights = []
    for __ in range(N_epoch):
        env.player_bursts = [0, 0]
        env.player_stand_lose = [0, 0]
        mean = 0.0
        for _ in range(N_eval):
            result = env.run_one_game(policy=basic_strategy)
            if _ == 0:
                print(result)
            mean += result[-1][2]
        mean /= N_eval
        # think reward mean as performance
        print(f"performance:{mean}")
    loses = [env.player_bursts, env.player_stand_lose]
    print(f"player_burst:{env.player_bursts[0]},{env.player_bursts[1]} player_stand_lose:{env.player_stand_lose[0]},{env.player_stand_lose[1]}")

    # # save result
    # print(rewards)
    # os.makedirs(data_dir, exist_ok=True)
    # os.makedirs(detail_dir, exist_ok=True)
    # now = datetime.datetime.now()
    # with open(os.path.join(data_dir, f"basic_lose.pkl"), "wb") as f:
    #     pickle.dump(loses, f)
    # with open(os.path.join(detail_dir, f"basic_lose_"+now.strftime('%Y%m%d_%H%M%S')+".pkl"), "wb") as f:
    #     pickle.dump(loses, f)
