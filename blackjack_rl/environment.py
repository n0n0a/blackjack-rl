import gym
from gym import spaces
from gym.utils import seeding
from typing import List, Tuple
from numpy.random import RandomState
from blackjack_rl.agent import Agent
from blackjack_rl.typedef import State, Action, Reward, Trans, Policy

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


# hand management class for blackjack
class Hand:
    # sum: sum of hand card numbers
    # have_eleven_ace: using eleven ace or not
    def __init__(self, sum: int = 0, open_card: int = None, ten_ace_count: int = 0, np_random: RandomState = None):
        assert 0 <= sum <= 20
        self.open_card = open_card
        self.sum = sum
        self.ten_ace_count = ten_ace_count
        self.deck = deck
        if np_random is None:
            self.np_random, _ = seeding.np_random(None)
        else:
            self.np_random = np_random

    # automatically change hand states on drawing card
    def draw(self, card: int = None) -> int:
        if card is None:
            card = self.np_random.choice(self.deck)
        assert 1 <= card <= 10
        self.sum += card
        if card == 1 and self.sum + 9 <= 21:
            self.ten_ace_count += 1
            self.sum += 9
        while self.sum > 21 and self.ten_ace_count > 0:
            self.ten_ace_count -= 1
            self.sum -= 9
        if self.open_card is None:
            self.open_card = card
        return card


class BlackjackEnv(gym.Env):
    # state consists of below 3 components
    # dealer_init: dealer initial hand
    # player_sum: player hand card sum
    # player_have_eleven_ace: whether the player has eleven ace or not
    def __init__(self, seed: int = None):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.deck = deck
        self.np_random = None
        self.player = Hand()
        self.dealer = Hand()
        self.seed(seed)
        self.reset()

    # set seed
    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        self.player.np_random = self.np_random
        self.dealer.np_random = self.np_random
        return [seed]

    # get observation(state)
    def _get_obs(self) -> State:
        assert self.dealer.open_card is not None
        return self.dealer.open_card, self.player.sum, self.player.ten_ace_count

    # set initial state
    def reset(self) -> State:
        self.player = Hand(np_random=self.np_random)
        self.dealer = Hand(np_random=self.np_random)
        self.player.draw()
        self.dealer.draw()
        return self._get_obs()

    # judge which wins on terminal
    def _judge_winner(self) -> int:
        dealer = Hand(sum=self.dealer.sum, ten_ace_count=self.dealer.ten_ace_count, np_random=self.np_random)
        while dealer.sum < 17:
            dealer.draw()
        if dealer.sum > 21:
            return 1
        else:
            return int(self.player.sum > dealer.sum) - int(self.player.sum < dealer.sum)

    # change state according to selected action
    def step(self, action: Action) -> Tuple[State, Reward, bool, dict]:
        done = False
        reward = 0
        if action:
            self.player.draw()
            if self.player.sum > 21:
                done = True
                reward = -1
            elif self.player.sum == 21:
                done = True
                reward = self._judge_winner()
        else:
            done = True
            reward = self._judge_winner()
        return self._get_obs(), reward, done, {}

    # make particular samples efficiently
    def make_samples(self, episode: int, agent: Agent = None) -> List[Trans]:
        samples = []
        for _ in range(episode):
            player = Hand(sum=self.np_random.choice(range(12, 21)),
                          open_card = self.np_random.choice(deck),
                          ten_ace_count=self.np_random.choice([2, 1, 0]),
                          np_random=self.np_random)
            dealer = Hand(np_random=self.np_random)
            dealer.draw()
            samples.extend(self.run_one_game(init_hand=(dealer, player), agent=agent))
        return samples

    # play one game and get game trajectory
    def run_one_game(self,
                     init_hand: Tuple[Hand, Hand] = None,
                     policy: Policy = None,
                     agent: Agent = None) -> List[Trans]:
        if init_hand is None:
            self.reset()
        else:
            self.dealer = init_hand[0]
            self.player = init_hand[1]
        if agent is not None:
            policy = agent.take_action
        if policy is None:
            policy = lambda s: self.np_random.choice([True, False])
        observation = self._get_obs()
        done = False
        reward = 0
        trajectory = []
        while not done:
            action = policy(observation)
            next_observation, reward, done, info = self.step(action)
            trajectory.append((observation, action, reward, next_observation))
            observation = next_observation
        return trajectory

    # not implemented
    def render(self, mode: str = 'human'):
        raise NotImplementedError
