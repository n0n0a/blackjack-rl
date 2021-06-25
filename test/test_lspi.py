from blackjack_rl.lspi import LSPIAgent
import numpy as np
from blackjack_rl.environment import BlackjackEnv


def test_constructor():
    agent = LSPIAgent()
    assert all(agent.weight == 0)


def test_reset_weight():
    agent = LSPIAgent()
    agent.weight = map(lambda x: x + np.random.randint(10), agent.weight)
    agent.reset_weight()
    assert all(agent.weight == 0)


def test_take_action():
    agent = LSPIAgent()

    # greedy
    assert agent.take_action(state=(0, 0, False))
    assert agent.take_action(state=(11, 11, True))

    # manually set weight
    state_case1 = (2, 12, False)
    action_case1 = False
    idx = agent._translate_weight_idx(state=agent._reindex_state(state_case1), action=action_case1)
    agent.weight[idx] = 1.0
    assert not agent.take_action(state=state_case1)
    action_case2 = True
    idx = agent._translate_weight_idx(state=agent._reindex_state(state_case1), action=action_case2)
    agent.weight[idx] = 2.0
    assert agent.take_action(state=state_case1)


def test_train():
    agent = LSPIAgent()

    # invalid samples
    old_weight = agent.weight
    samples = [((12, 12, False), False, 0, (12, 12, False)),
               ((11, 5, True), False, 0, (21, 21, False)),
               ((12, 6, False), False, 0, (1, 1, False)),
               ((21, 21, True), False, 0, (2, 10, False))]
    old_samples = samples.copy()
    agent.train(train_data=samples)
    assert agent.weight.shape == old_weight.shape
    assert all(agent.weight == 0)

    # valid samples
    env = BlackjackEnv(seed=0)
    samples = env.make_samples(episode=100)
    agent.train(samples)
    assert any(agent.weight != 0)


def test_calculate_A():
    agent = LSPIAgent()

    # one sample
    old_weight = agent.weight
    samples = [((0, 0, False), False, 0, (2, 12, False))]
    A = agent._calculate_A(train_data=samples)
    A = A.toarray()
    assert np.count_nonzero(A > 0.001) == 1
    assert abs(A[0, 0] - 1) < 0.001
    assert np.count_nonzero(A < -0.001) == 1
    assert abs(A[0, 180] + 1) < 0.001

    # one more sample
    samples.append(((3, 4, True), True, 0, (11, 19, True)))
    A = agent._calculate_A(train_data=samples)
    A = A.toarray()
    assert np.count_nonzero(A > 0.001) == 2
    assert abs(A[0, 0] - 1) < 0.001
    assert abs(A[313, 313] - 1) < 0.001
    assert np.count_nonzero(A < -0.001) == 2
    assert abs(A[0, 180] + 1) < 0.001
    assert abs(A[313, 349] + 1) < 0.001


def test_calculate_b():
    agent = LSPIAgent()

    # one sample
    old_weight = agent.weight
    samples = [((0, 0, False), False, 1, (2, 12, False))]
    b = agent._calculate_b(train_data=samples)
    b = b.toarray()
    assert np.count_nonzero(b != 0) == 1
    assert abs(b[0, 0] - 1) < 0.001

    # one more sample
    samples.append(((3, 4, True), True, -1, (11, 19, True)))
    b = agent._calculate_b(train_data=samples)
    b = b.toarray()
    assert np.count_nonzero(b != 0) == 2
    assert abs(b[0, 0] - 1) < 0.001
    assert abs(b[313, 0] + 1) < 0.001


def test_isvalid():
    agent = LSPIAgent()

    # samples given by env
    env = BlackjackEnv(seed=0)
    samples = env.make_samples(episode=1000)
    assert all(agent._isvalid(s[0]) for s in samples)


def test_reindex_state():
    agent = LSPIAgent()

    # easy case
    sample = (2, 14, True)
    assert agent._reindex_state(sample) == (0, 2, True)
    sample = (11, 20, False)
    assert agent._reindex_state(sample) == (9, 8, False)


def _translate_weight_idx():
    agent = LSPIAgent()

    # easy case
    sample = (0, 2, True)
    assert agent._reindex_state(sample, False) == 110
    sample = (9, 8, False)
    assert agent._reindex_state(sample, True) == 269
    pass