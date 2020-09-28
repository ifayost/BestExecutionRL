import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RL.Environments import MarketGym
from RL.Rewards import mean_step
from RL.Agents import DQN

plt.style.use("solarized_dark")

PATH = "./episodes"

env = MarketGym(PATH, mean_step)

state = env.reset()

weights = "./weights/DDQN_mean_step.pt"

alpha = 1e-2  # 5e-4
gamma = 0.999
epsilon = 0.25

episodes = 4000  # len(env.episodes)
batch_size = 64
target_update = 4

discount = 1-1/(episodes*0.15)


def adaptive(self, episode):
    self.epsilon = max(0.01, min(1.0, self.epsilon*discount))


agent = DQN(env, alpha, gamma, epsilon, adaptive=adaptive,
            double=True, save=weights)

stats = agent.train(env, episodes, batch_size, target_update)
checks = np.array(stats['checkpoints']).astype(int)
rewards = np.array(stats['rewards'])
smooth = pd.DataFrame(rewards).rolling(50).mean()
plt.rcParams['figure.figsize'] = (20, 10)
plt.plot(range(len(rewards)), rewards, alpha=0.5)
plt.plot(range(len(smooth)), smooth)
plt.scatter(checks, smooth.iloc[checks], c='r', marker='.')
inf, sup = np.quantile(rewards, [0.05, 0.95])
plt.ylim(inf, sup)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.grid(True)
plt.savefig(weights[:-2]+'png', dpi=200)
plt.show()
