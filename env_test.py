import torch
import matplotlib.pyplot as plt
from RL.Environments import MarketGym
from RL.Agents import TWAP, POV, DQN, dynamicTWAP
from RL.Rewards import mean_step
from RL.Utils import test_episode_agent

plt.style.use("solarized_dark")

PATH = "./episodes"
env = MarketGym(PATH, mean_step)

weights_dqn = "./weights/DDQN_mean_step.pt"

alpha = 5e-4
gamma = 0.999
epsilon = 0.1


dqn = DQN(env, alpha, gamma, epsilon,
          double=True, save=weights_dqn)
dqn.Q_net.load_state_dict(torch.load(weights_dqn))


random_seed = 5

test_episode_agent(env, dqn, name="DQN", random_seed=random_seed)

twap = TWAP()
twap.train(env)
test_episode_agent(env, twap, name="TWAP", random_seed=random_seed)

dtwap = dynamicTWAP()
dtwap.train(env)
test_episode_agent(env, dtwap, name="dynamicTWAP", random_seed=random_seed)

pov = POV(0.1)
test_episode_agent(env, pov, name="POV", random_seed=random_seed, POV=True)
