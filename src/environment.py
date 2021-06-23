import gym
from gym import spaces
from gym.utils import seeding
from typing import Tuple, List
from numpy.random import RandomState


class Hand:
    def __init__(self, sum: int = 0, have_eleven_ace: bool = False, np_random: RandomState = None):
        self.sum = sum
        self.have_eleven_ace = have_eleven_ace
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.np_random = np_random

    def draw(self) -> int:
        card = self.np_random.choice(self.deck)
        self.sum += card
        if card == 1 and self.have_eleven_ace == False and self.sum + 10 <= 21:
            self.have_eleven_ace = True
            self.sum += 10
        elif self.sum > 21 and self.have_eleven_ace == True:
            self.have_eleven_ace = False
            self.sum -= 10
        return card


class BlackjackEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.np_random = None
        self.player = Hand()
        self.dealer = Hand()
        self.reset()

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        self.player.np_random = self.np_random
        self.dealer.np_random = self.np_random
        return [seed]

    def _get_obs(self) -> Tuple[int, int, bool]:
        return self.player.sum, self.dealer.sum, self.player.have_eleven_ace

    def reset(self) -> Tuple[int, int, bool]:
        self.seed()
        self.player.draw()
        self.dealer.draw()
        return self._get_obs()

    def _judge_winner(self) -> int:
        while self.dealer.sum < 17:
            self.dealer.draw()
        if self.dealer.sum > 21:
            return 1
        else:
            return int((self.player.sum > self.dealer.sum) - (self.player.sum < self.dealer.sum))

    def step(self, action: bool) -> Tuple[Tuple[int, int, bool], int, bool, dict]:
        assert self.action_space.contains(action)
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
            self.dealer = Hand().draw()
            self.player.sum = self.np_random.choice(range(12, 21))
            self.player.have_eleven_ace = self.np_random.choice([True, False])
            observation = self._get_obs()
            done = False
            reward = 0
            while not done:
                action = self.np_random.choice([True, False])
                next_observation, reward, done, info = self.step(action)
                samples.append((observation, action, reward, next_observation))
        return samples
