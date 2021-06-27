from blackjack_rl.environment import Hand
from gym.utils import seeding


def test_constructor():
    hand = Hand()
    assert hand.sum == 0
    assert hand.ten_ace_count == 0
    hand = Hand(sum=10, ten_ace_count=1)
    assert hand.sum == 10
    assert hand.ten_ace_count == 1


def test_draw():
    np_random, seed = seeding.np_random(5)

    # 10 False Start
    hand = Hand(sum=10, ten_ace_count=0)
    assert hand.sum == 10
    assert hand.ten_ace_count == 0
    hand.draw(card=1)
    assert hand.sum == 20
    assert hand.ten_ace_count == 1
    hand.draw(card=1)
    assert hand.sum == 21
    assert hand.ten_ace_count == 1
    hand.draw(card=1)
    assert hand.sum == 13
    assert hand.ten_ace_count == 0

    # 11 True Start
    hand = Hand(sum=11, ten_ace_count=1)
    assert hand.sum == 11
    assert hand.ten_ace_count == 1
    hand.draw(card=1)
    assert hand.sum == 21
    assert hand.ten_ace_count == 2
    hand.draw(card=10)
    assert hand.sum == 13
    assert hand.ten_ace_count == 0
    hand.draw(card=1)
    assert hand.sum == 14
    assert hand.ten_ace_count == 0
    hand.draw(card=10)
    assert hand.sum == 24
    assert hand.ten_ace_count == 0

    # 11 False Start
    hand = Hand(sum=11, ten_ace_count=0)
    assert hand.sum == 11
    assert hand.ten_ace_count == 0
    hand.draw(card=1)
    assert hand.sum == 21
    assert hand.ten_ace_count == 1
    hand.draw(card=10)
    assert hand.sum == 22
    assert hand.ten_ace_count == 0

    # np_random test
    hand = Hand(np_random=np_random)
    hand.draw()
    assert hand.sum == 10
    assert hand.ten_ace_count == 0
    hand.draw()
    assert hand.sum == 18
    assert hand.ten_ace_count == 0
    hand.draw()
    assert hand.sum == 25
    assert hand.ten_ace_count == 0
