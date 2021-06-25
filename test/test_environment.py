from blackjack_rl.environment import BlackjackEnv, Hand


def test_constructor():
    env = BlackjackEnv(seed=1)
    assert env.player.sum == 11
    assert env.player.have_eleven_ace
    assert env.dealer.sum == 5
    assert not env.dealer.have_eleven_ace


def test_seed():
    env = BlackjackEnv(seed=2)
    assert env.np_random.choice(range(1000)) == 877
    assert env.np_random.randint(1000) == 302


def test_reset():
    env = BlackjackEnv(seed=3)
    observation = env.reset()
    assert env.player.sum == 4
    assert not env.player.have_eleven_ace
    assert env.dealer.sum == 2
    assert not env.player.have_eleven_ace
    assert observation == (2, 4, False)


def test_judge_winner():
    env = BlackjackEnv(seed=4)

    # lose
    assert env._judge_winner() == -1
    assert env.dealer.sum == 19

    # win
    env.reset()
    assert env._judge_winner() == 1
    assert env.dealer.sum == 23

    # draw
    env.reset()
    env.player.sum = 18
    assert env._judge_winner() == 0
    assert env.dealer.sum == 18


def test_step():
    env = BlackjackEnv(seed=5)

    # 1st trial
    observation, reward, done, info = env.step(True)
    assert observation == (8, 17, False)
    assert reward == 0
    assert not done
    assert info == {}
    observation, reward, done, info = env.step(True)
    assert observation == (8, 27, False)
    assert reward == -1
    assert done
    assert info == {}

    # 2nd trial
    env.reset()
    observation, reward, done, info = env.step(True)
    assert observation == (8, 19, True)
    assert reward == 0
    assert not done
    assert info == {}
    observation, reward, done, info = env.step(True)
    assert observation == (8, 18, False)
    assert reward == 0
    assert not done
    assert info == {}
    observation, reward, done, info = env.step(False)
    assert observation == (17, 18, False)
    assert reward == 1
    assert done
    assert info == {}

    # 3rd trial
    env.reset()
    observation, reward, done, info = env.step(True)
    assert observation == (3, 16, False)
    assert reward == 0
    assert not done
    assert info == {}
    observation, reward, done, info = env.step(True)
    assert observation == (3, 23, False)
    assert reward == -1
    assert done
    assert info == {}


def test_make_samples():
    env = BlackjackEnv(seed=6)
    samples = env.make_samples(episode=10000)
    assert not [x[0] for x in samples
                if (x[0][0] < 2 or 11 < x[0][0])
                and (x[0][1] < 12 or 20 < x[0][1])
                and (not x[0][2] in [True, False])]


def test_run_one_game():
    env = BlackjackEnv(seed=7)
    trajectory = env.run_one_game()
    assert not [x[0] for x in trajectory
                if (x[0][0] < 2 or 11 < x[0][0])
                and (x[0][1] < 12 or 20 < x[0][1])
                and (not x[0][2] in [True, False])]
    trajectory = env.run_one_game(init_hand=(Hand(sum=20, np_random=env.np_random), Hand(sum=2, np_random=env.np_random)))
    assert not [x[0] for x in trajectory
                if (x[0][0] < 2 or 11 < x[0][0])
                and (x[0][1] < 12 or 20 < x[0][1])
                and (not x[0][2] in [True, False])]
