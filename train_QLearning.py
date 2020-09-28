import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RL.Environments import MarketGym
from RL.Rewards import simple_rewards
from RL.Agents import QLearning

plt.style.use("solarized_dark")

PATH = "../DATA/SAN/san_orderbooks.csv"

HORIZON = pd.to_timedelta(5, unit='min')
INVENTORY = 300_000
TIME_DELTA = pd.to_timedelta(0.5, unit='s')
env = MarketGym(PATH, HORIZON, INVENTORY, simple_rewards, TIME_DELTA)

state = env.reset()

weights = "./weights/QLearning-Environment.npy"

alpha = 0.1  # 0.4
gamma = 0.999
epsilon = 0.5

episodes = 3000

discount = 1-1/(episodes*0.15)


def adaptive(self, episode):
    self.epsilon = max(0.01, min(1.0, self.epsilon*discount))


OBSERVATION_DIMS = env.observation_space.shape[0]
RESOLUTIONS = [20, 20]
ALL_POSSIBLE_STATES = np.array(np.meshgrid(
    *[range(res) for res in RESOLUTIONS])).T.reshape(-1, OBSERVATION_DIMS)
STATE_SPACE = {tuple(j): i for i, j in enumerate(ALL_POSSIBLE_STATES)}
LOWER = env.observation_space.low
HIGHER = env.observation_space.high


def discretize(state):
    for i in range(OBSERVATION_DIMS):
        state[i] = np.digitize(state[i],
                               np.linspace(LOWER[i],
                                           HIGHER[i],
                                           RESOLUTIONS[i]-1))
    return STATE_SPACE[tuple(state.astype(int))]


agent = QLearning(alpha, gamma, epsilon, adaptive=adaptive,
                  discretize=discretize, double=True, save=weights)


stats = agent.train(env, episodes)
checks = np.array(stats['checkpoints']).astype(int)
rewards = np.array(stats['rewards'])
smooth = pd.DataFrame(rewards).rolling(40).mean()
plt.rcParams['figure.figsize'] = (20, 20)
plt.plot(range(len(rewards)), rewards, alpha=0.5)
plt.plot(range(len(smooth)), smooth)
plt.scatter(checks, smooth.iloc[checks], c='r', marker='.')
inf, sup = np.quantile(rewards, [0.01, 0.99])
plt.ylim(inf, sup)
plt.grid(True)
plt.savefig(weights[:-3]+'png', dpi=200)
plt.show()
