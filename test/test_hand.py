from blackjack_rl.environment import Hand
from gym.utils import seeding


def test_constructor():
    hand = Hand()
    assert hand.sum == 0
    assert not hand.have_eleven_ace
    hand = Hand(sum=10, have_eleven_ace=True)
    assert hand.sum == 10
    assert hand.have_eleven_ace


def test_draw():
    np_random, seed = seeding.np_random(5)
    hand = Hand(sum=10, have_eleven_ace=False, np_random=np_random)
    hand.draw()