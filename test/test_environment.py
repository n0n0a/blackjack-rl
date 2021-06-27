from blackjack_rl.environment import BlackjackEnv, Hand
from blackjack_rl.lspi import LSPIAgent


def test_constructor():
    env = BlackjackEnv(seed=1)
    assert env.player.sum == 10
    assert env.player.ten_ace_count == 1
    assert env.dealer.sum == 5
    assert env.dealer.ten_ace_count == 0


def test_seed():
    env = BlackjackEnv(seed=2)
    assert env.np_random.choice(range(1000)) == 877
    assert env.np_random.randint(1000) == 302


def test_reset():
    env = BlackjackEnv(seed=3)
    observation = env.reset()
    assert env.player.sum == 4
    assert env.player.ten_ace_count == 0
    assert env.dealer.sum == 2
    assert env.player.ten_ace_count == 0
    assert observation == (2, 4, 0)


def test_judge_winner():
    env = BlackjackEnv(seed=4)

    # lose
    assert env._judge_winner() == -1
    assert env.dealer.sum == 10

    # win
    env.reset()
    assert env._judge_winner() == 1
    assert env.dealer.sum == 10

    # draw
    env.reset()
    env.player.sum = 18
    assert env._judge_winner() == 0
    assert env.dealer.sum == 8


def test_step():
    env = BlackjackEnv(seed=5)

    # 1st trial
    observation, reward, done, info = env.step(True)
    assert observation == (8, 17, 0)
    assert reward == 0
    assert not done
    assert info == {}
    observation, reward, done, info = env.step(True)
    assert observation == (8, 27, 0)
    assert reward == -1
    assert done
    assert info == {}

    # 2nd trial
    env.reset()
    observation, reward, done, info = env.step(True)
    assert observation == (8, 18, 1)
    assert reward == 0
    assert not done
    assert info == {}
    observation, reward, done, info = env.step(True)
    assert observation == (8, 18, 0)
    assert reward == 0
    assert not done
    assert info == {}
    observation, reward, done, info = env.step(False)
    assert observation == (8, 18, 0)
    assert reward == 1
    assert done
    assert info == {}

    # 3rd trial
    env.reset()
    observation, reward, done, info = env.step(True)
    assert observation == (3, 16, 0)
    assert reward == 0
    assert not done
    assert info == {}
    observation, reward, done, info = env.step(True)
    assert observation == (3, 23, 0)
    assert reward == -1
    assert done
    assert info == {}


def test_make_samples():
    env = BlackjackEnv(seed=6)
    samples = env.make_samples(episode=10000)
    assert all(x[0] for x in samples
                if (1 <= x[0][0] <= 10)
                and (12 <= x[0][1] <= 20)
                and (x[0][2] in [0, 1, 2]))


def test_run_one_game():
    env = BlackjackEnv(seed=7)

    # policy: random
    trajectory = env.run_one_game()
    assert all(x for x in trajectory
               if (1 <= x[0][0] <= 10)
               and (1 <= x[0][1] <= 20)
               and (x[0][2] in [0, 1, 2]))
    trajectory = env.run_one_game(init_hand=(Hand(sum=2, open_card=2, np_random=env.np_random), Hand(sum=20, np_random=env.np_random)))
    assert all(x[0] for x in trajectory
               if (1 <= x[0][0] <= 10)
               and (12 <= x[0][1] <= 20)
               and (x[0][2] in [0, 1, 2]))

    # policy: draw only if sum < 17
    policy = lambda s: s[1] < 17
    trajectory = env.run_one_game(policy=policy)
    assert all(x for x in trajectory
               if (1 <= x[0][0] <= 10)
               and (1 <= x[0][1] <= 20)
               and (x[0][2] in [0, 1, 2]))
    trajectory = env.run_one_game(init_hand=(Hand(sum=2, open_card=2, np_random=env.np_random),
                                             Hand(sum=20, np_random=env.np_random)),
                                  policy=policy)
    assert all(x[0] for x in trajectory
               if (1 <= x[0][0] <= 10)
               and (12 <= x[0][1] <= 20)
               and (x[0][2] in [0, 1, 2]))

    # policy: LSPIAgent
    agent = LSPIAgent()
    samples = env.make_samples(episode=100)
    agent.train(train_data=samples)
    trajectory = env.run_one_game(agent=agent)
    assert all(x for x in trajectory
               if (1 <= x[0][0] <= 10)
               and (1 <= x[0][1] <= 20)
               and (x[0][2] in [0, 1, 2]))
    trajectory = env.run_one_game(init_hand=(Hand(sum=2, open_card=2, np_random=env.np_random),
                                             Hand(sum=20, np_random=env.np_random)),
                                  policy=policy)
    assert all(x[0] for x in trajectory
               if (1 <= x[0][0] <= 10)
               and (12 <= x[0][1] <= 20)
               and (x[0][2] in [0, 1, 2]))
