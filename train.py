import matplotlib.pyplot as plt
import pickle
import json
import pandas as pd
from RL.Environment_v2 import MarketGym, make_variables
from RL.Rewards import vwap_reward
from RL.Agents import DQN
from RL.Utils import plot_train_stats

plt.style.use('./solarized_dark.mplstyle')

PATH = "/Users/ifayost/Desktop/TFM/Projects/DATA/SAN_month/"\
    "best_exec_data/orderbook"

timestamp = str(
    pd.Timestamp.now()
    ).replace(' ', '_').replace(':', '-').split('.')

H = pd.to_timedelta(1, unit='hour')
V = 500_000
buy = True
time_step = pd.to_timedelta(1, unit='s')
make_variables = make_variables

env = MarketGym(PATH, H, V, buy, time_step, make_variables, vwap_reward)

state = env.reset()

weights = f'./weights/DDQN_{timestamp}.pt'

alpha = 1e-2  # 5e-4
gamma = 0.999
epsilon = 0.9
double = True

episodes = 10
batch_size = 64
target_update = 4

discount = 1-1/(episodes*0.15)


def adaptive(self, episode):
    self.epsilon = max(0.01, min(1.0, self.epsilon*discount))


agent = DQN(env, alpha, gamma, epsilon, adaptive=adaptive,
            double=double, save=weights, rewards_mean=100,
            n_episodes_to_save=50)

info = {'horizon': str(env.H), 'volume': str(env.V),
        'buy': str(env.buy), 'time_step': str(env.time_step),
        'reward_function': 'vwap_reward',
        'algo': {'name': 'DDQN', 'alpha': alpha, 'gamma': gamma,
                 'epsilon_ini': epsilon, 'discoung': discount,
                 'double': double},
        'training': {'episodes': episodes, 'batch_size': batch_size,
                     'target_update': target_update}}
with open(f'./weights/DDQN_{timestamp}.txt', 'w') as f:
    f.write(json.dumps(info))

stats = agent.train(env, episodes, batch_size, target_update)
with open(f'./figures/DDQN_{timestamp}.pkl', 'wb') as f:
    pickle.dump(stats, f)

plot_train_stats(stats, save=f'./figures/DDQN_{timestamp}', rolling=100)
