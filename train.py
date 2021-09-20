import pickle
import pandas as pd
import json
from RL.Environment import MarketGym
from RL.Rewards import vwap_reward_penalty
from RL.Agents import DQN
from RL.Utils import plot_train_stats


PATH = "./episodes/simple"

timestamp = str(
    pd.Timestamp.now()
    ).replace(' ', '_').replace(':', '-').split('.')[0]

reward_penalty = 100_000
env = MarketGym(PATH, vwap_reward_penalty(reward_penalty))

state = env.reset()

model_name = 'DDQN'
weights = f'./weights/{model_name}_{timestamp}.pt'

alpha = 5e-4
gamma = 0.99
epsilon = 1

episodes = 10000
batch_size = 64
target_update = 4

discount = 1-1/(episodes*0.2)
epsilon_min = 0.05


def adaptive(self, episode):
    self.epsilon = max(epsilon_min, min(1.0, self.epsilon*discount))


agent = DQN(env, alpha, gamma, epsilon, adaptive=adaptive,
            double=True, save=weights, rewards_mean=100,
            n_episodes_to_save=50)


info = {'horizon': str(env.H), 'volume': str(env.V),
        'buy': str(env.buy), 'time_step': str(env.time_step),
        'PATH': PATH, 'variables': str(env.variables),
        'reward_function': 'vwap_reward_penalty',
        'reward_penalty': reward_penalty,
        'algo': {'name': model_name, 'alpha': alpha, 'gamma': gamma,
                 'epsilon_ini': epsilon, 'epsilon_min': epsilon_min,
                 'discount': discount,
                 'double': True},
        'training': {'episodes': episodes, 'batch_size': batch_size,
                     'target_update': target_update}}

with open(f'./weights/info_{model_name}_{timestamp}.txt', 'w') as f:
    f.write(json.dumps(info))

stats = agent.train(env, episodes, batch_size, target_update)
with open(f'./figures/{model_name}_{timestamp}.pkl', 'wb') as f:
    pickle.dump(stats, f)

plot_train_stats(stats, save=f'./figures/{model_name}_{timestamp}', rolling=50)
