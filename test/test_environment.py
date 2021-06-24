from blackjack_rl.environment import BlackjackEnv


def test_constructor():
    env = BlackjackEnv()
    assert env.player.sum in range(1, 12)
    assert env.dealer.sum in range(1, 12)


def test_reset():
    env = BlackjackEnv()
    env.reset()
    assert env.player.sum in range(1, 12)
    assert env.dealer.sum in range(1, 12)
