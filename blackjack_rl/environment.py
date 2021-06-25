import gym
from gym import spaces
from gym.utils import seeding
from typing import Tuple, List
from numpy.random import RandomState

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


class Hand:
    def __init__(self, sum: int = 0, have_eleven_ace: bool = False, np_random: RandomState = None):
        self.sum = sum
        self.have_eleven_ace = have_eleven_ace
        self.deck = deck
        if np_random is None:
            self.np_random, _ = seeding.np_random(None)
        else:
            self.np_random = np_random

    def draw(self, card: int = None) -> int:
        if card is None:
            card = self.np_random.choice(self.deck)
        self.sum += card
        if card == 1 and (not self.have_eleven_ace) and self.sum + 10 <= 21:
            self.have_eleven_ace = True
            self.sum += 10
        elif self.sum > 21 and self.have_eleven_ace:
            self.have_eleven_ace = False
            self.sum -= 10
        return card


class BlackjackEnv(gym.Env):
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

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        self.player.np_random = self.np_random
        self.dealer.np_random = self.np_random
        return [seed]

    def _get_obs(self) -> Tuple[int, int, bool]:
        return self.dealer.sum, self.player.sum, self.player.have_eleven_ace

    def reset(self) -> Tuple[int, int, bool]:
        self.player = Hand(np_random=self.np_random)
        self.dealer = Hand(np_random=self.np_random)
        self.player.draw()
        self.dealer.draw()
        return self._get_obs()

    def _judge_winner(self) -> int:
        while self.dealer.sum < 17:
            self.dealer.draw()
        if self.dealer.sum > 21:
            return 1
        else:
            return int(self.player.sum > self.dealer.sum) - int(self.player.sum < self.dealer.sum)

    def step(self, action: bool) -> Tuple[Tuple[int, int, bool], int, bool, dict]:
        assert self.action_space.contains(int(action))
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

    def make_samples(self, episode: int) -> List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]:
        samples = []
        for _ in range(episode):
            player = Hand(sum=self.np_random.choice(range(12, 21)),
                          have_eleven_ace=self.np_random.choice([True, False]),
                          np_random=self.np_random)
            dealer_draw = self.np_random.choice(range(2, 11))
            dealer = Hand(sum=dealer_draw,
                          have_eleven_ace=dealer_draw==11,
                          np_random=self.np_random)
            samples.extend(self.run_one_game(init_hand=(player, dealer)))
        return samples

    def run_one_game(self, init_hand: Tuple[Hand, Hand] = None) -> List[Tuple[Tuple[int, int, bool], bool, int, Tuple[int, int, bool]]]:
        if init_hand is None:
            self.reset()
        else:
            self.player = init_hand[0]
            self.dealer = init_hand[1]
        observation = self._get_obs()
        done = False
        reward = 0
        trajectory = []
        while not done:
            action = self.np_random.choice([True, False])
            next_observation, reward, done, info = self.step(action)
            trajectory.append((observation, action, reward, next_observation))
        return trajectory

    def render(self, mode='human'):
        raise NotImplementedError
