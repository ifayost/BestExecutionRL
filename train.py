import matplotlib.pyplot as plt
import pickle
from RL.Environment import MarketGym
from RL.Rewards import vwap_reward
from RL.Agents import DQN
from RL.Utils import plot_train_stats

plt.style.use('./solarized_dark.mplstyle')

PATH = "./episodes"

env = MarketGym(PATH, vwap_reward)

state = env.reset()

weights = "./weights/DDQN.pt"

alpha = 1e-2  # 5e-4
gamma = 0.999
epsilon = 0.9

episodes = 8000
batch_size = 64
target_update = 4

discount = 1-1/(episodes*0.15)


def adaptive(self, episode):
    self.epsilon = max(0.01, min(1.0, self.epsilon*discount))


agent = DQN(env, alpha, gamma, epsilon, adaptive=adaptive,
            double=True, save=weights, rewards_mean=100,
	    n_episodes_to_save=50)

stats = agent.train(env, episodes, batch_size, target_update)
with open('./figures/DDQN.pkl', 'wb') as f:
    pickle.dump(stats, f)

plot_train_stats(stats, save='./figures/DDQN', rolling=100)
