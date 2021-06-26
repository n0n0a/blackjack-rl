import time
from blackjack_rl.environment import BlackjackEnv
from blackjack_rl.lspi import LSPIAgent

N_episode = 100000
N_train = 1
N_run = 100000


def measure_process_time():
    env = BlackjackEnv(seed=0)
    agent = LSPIAgent()

    # time of make_samples
    start = time.time()
    samples = env.make_samples(episode=N_episode)
    print(f"make_sample(): {time.time()-start} ms")

    # time of train
    mini_batch = samples[:N_episode//10]
    start = time.time()
    for _ in range(N_train):
        agent.train(train_data=mini_batch)
    print(f"make_sample(): {time.time()-start} ms")

    # time of run_game
    start = time.time()
    for t in range(N_run):
        env.run_one_game(agent=agent)
    print(f"make_sample(): {time.time()-start} ms")


if __name__ == '__main__':
    measure_process_time()
