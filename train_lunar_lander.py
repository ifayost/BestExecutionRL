import gym
import pickle
import pandas as pd
import json
from RL.Agents import DQN
from RL.Utils import plot_train_stats


timestamp = str(
    pd.Timestamp.now()
    ).replace(' ', '_').replace(':', '-').split('.')[0]

env = gym.make("LunarLander-v2")

state = env.reset()

model_name = 'DDQN_LL'
weights = f'./weights/{model_name}_{timestamp}.pt'

alpha = 1e-2  # 5e-4
gamma = 0.999
epsilon = 0.9

episodes = 10000
batch_size = 128
target_update = 10

discount = 1-1/(episodes*0.4)
epsilon_min = 0.05


def adaptive(self, episode):
    self.epsilon = max(epsilon_min, min(1.0, self.epsilon*discount))


agent = DQN(env, alpha, gamma, epsilon,  adaptive=adaptive,
            double=True, save=weights, rewards_mean=100,
            n_episodes_to_save=50)


info = {'algo': {'name': model_name, 'alpha': alpha, 'gamma': gamma,
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
