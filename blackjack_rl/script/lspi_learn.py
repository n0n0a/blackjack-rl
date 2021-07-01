from blackjack_rl.envs.eleven_ace import BlackjackEnv
from blackjack_rl.agent.lspi import LSPIAgent
import os, pickle

# environment seed
seed = 3
# make_sample episode count
N_epoch = 10000
# LSPI train count
N_episode = 100
# Evaluation count per leaning
N_eval = 10000
# data dir
data_dir = "../../data"


if __name__ == '__main__':
    # initialize
    env = BlackjackEnv(seed=seed)
    agent = LSPIAgent()

    # prepare training data
    samples = env.make_samples(episode=N_episode)

    # learning
    rewards = []
    for epoch in range(N_epoch):
        updated = agent.train(train_data=samples)
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
        if not updated:
            print("resampling...")
            samples = env.make_samples(episode=N_episode, agent=agent)

    # save result
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "lspi_rewards.txt"), "wb") as f:
        pickle.dump(rewards, f)
