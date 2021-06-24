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
    # 10 False Start
    hand = Hand(sum=10, have_eleven_ace=False, np_random=np_random)
    assert hand.sum == 10
    assert not hand.have_eleven_ace
    hand.draw(card=1)
    assert hand.sum == 21
    assert hand.have_eleven_ace
    hand.draw(card=1)
    assert hand.sum == 12
    assert not hand.have_eleven_ace
    hand.draw(card=1)
    assert hand.sum == 13
    assert not hand.have_eleven_ace
    # 11 True Start
    hand = Hand(sum=11, have_eleven_ace=True, np_random=np_random)
    assert hand.sum == 11
    assert hand.have_eleven_ace
    hand.draw(card=1)
    assert hand.sum == 12
    assert hand.have_eleven_ace
    hand.draw(card=10)
    assert hand.sum == 12
    assert not hand.have_eleven_ace
    hand.draw(card=1)
    assert hand.sum == 13
    assert not hand.have_eleven_ace
    hand.draw(card=10)
    assert hand.sum == 23
    assert not hand.have_eleven_ace
    # 11 False Start
    hand = Hand(sum=11, have_eleven_ace=False, np_random=np_random)
    assert hand.sum == 11
    assert not hand.have_eleven_ace
    hand.draw(card=1)
    assert hand.sum == 12
    assert not hand.have_eleven_ace
    hand.draw(card=10)
    assert hand.sum == 22
    assert not hand.have_eleven_ace
