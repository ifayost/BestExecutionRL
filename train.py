import matplotlib.pyplot as plt
from RL.Environment import MarketGym
from RL.Rewards import vwap_reward
from RL.Agents import DQN
from RL.Utils import plot_train_stats

plt.style.use("solarized_dark")

PATH = "./episodes"

env = MarketGym(PATH, vwap_reward)

state = env.reset()

weights = "./weights/DDQN.pt"

alpha = 1e-2  # 5e-4
gamma = 0.999
epsilon = 0.9

episodes = 10
batch_size = 64
target_update = 4

discount = 1-1/(episodes*0.15)


def adaptive(self, episode):
    self.epsilon = max(0.01, min(1.0, self.epsilon*discount))


agent = DQN(env, alpha, gamma, epsilon, adaptive=adaptive,
            double=True, save=weights)

stats = agent.train(env, episodes, batch_size, target_update)

plot_train_stats(stats, save='./figura', rolling=2)
