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
    def __init__(self, sum: int = 0, have_eleven_ace: bool = False, np_random: RandomState = None):
        assert 0 <= sum <= 20
        self.sum = sum
        self.have_eleven_ace = have_eleven_ace
        self.deck = deck
        if np_random is None:
            self.np_random, _ = seeding.np_random(None)
        else:
            self.np_random = np_random

    # automatically change hand states on drawing card
    def draw(self, card: int = None) -> int:
        if card is None:
            card = self.np_random.choice(self.deck)
        assert 0 <= card <= 10
        self.sum += card
        if card == 1 and (not self.have_eleven_ace) and self.sum + 10 <= 21:
            self.have_eleven_ace = True
            self.sum += 10
        elif self.sum > 21 and self.have_eleven_ace:
            self.have_eleven_ace = False
            self.sum -= 10
        return card


class BlackjackEnv(gym.Env):
    # state consists of below 3 components
    # dealer_sum: dealer hand card sum
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
        return self.dealer.sum, self.player.sum, self.player.have_eleven_ace

    # set initial state
    def reset(self) -> State:
        self.player = Hand(np_random=self.np_random)
        self.dealer = Hand(np_random=self.np_random)
        self.player.draw()
        self.dealer.draw()
        return self._get_obs()

    # judge which wins on terminal
    def _judge_winner(self) -> int:
        dealer = Hand(sum=self.dealer.sum, have_eleven_ace=self.dealer.have_eleven_ace, np_random=self.np_random)
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
    def make_samples(self, episode: int) -> List[Trans]:
        samples = []
        for _ in range(episode):
            player = Hand(sum=self.np_random.choice(range(12, 21)),
                          have_eleven_ace=self.np_random.choice([True, False]),
                          np_random=self.np_random)
            dealer_draw = self.np_random.choice(range(2, 11))
            dealer = Hand(sum=dealer_draw,
                          have_eleven_ace=(dealer_draw == 11),
                          np_random=self.np_random)
            samples.extend(self.run_one_game(init_hand=(player, dealer)))
        return samples

    # play one game and get game trajectory
    def run_one_game(self,
                     init_hand: Tuple[Hand, Hand] = None,
                     policy: Policy = None,
                     agent: Agent = None) -> List[Trans]:
        if init_hand is None:
            self.reset()
        else:
            self.player = init_hand[0]
            self.dealer = init_hand[1]
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
