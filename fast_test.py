# import torch
import matplotlib.pyplot as plt
from RL.Environment import MarketGym, free_step
from RL.Agents import TWAP, POV, DQN
from RL.Rewards import vwap_reward
from RL.Utils import test_episode_agent

plt.style.use("solarized_dark")

PATH = "./episodes"
env = MarketGym(PATH, vwap_reward)

# weights_dqn = "./weights/DDQN_mean_step.pt"
alpha = 5e-4
gamma = 0.999
epsilon = 0.1
dqn = DQN(env, alpha, gamma, epsilon,
          double=True)  # , save=weights_dqn)
# dqn.Q_net.load_state_dict(torch.load(weights_dqn))


random_seed = 1

test_episode_agent(env, dqn, name="DQN", random_seed=random_seed)

env.step = free_step(env)

twap = TWAP()
twap.train(env)
test_episode_agent(env, twap, name="TWAP", random_seed=random_seed)

pov = POV(0.1)
test_episode_agent(env, pov, name="POV",
                   random_seed=random_seed, episode_retrain=True)
