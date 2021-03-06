# import torch
import matplotlib.pyplot as plt
import torch
from RL.Environment import MarketGym, free_step
from RL.Agents import TWAP, POV, DQN
from RL.Rewards import vwap_reward
from RL.Utils import test_agent

plt.style.use("./solarized_dark.mplstyle")

EPISODES = "./episodes"
TEST = "./test"

env = MarketGym(EPISODES, vwap_reward, mode='test')

alpha = 5e-4
gamma = 0.999
epsilon = 0.1

weights_dqn = "./weights/DDQN.pt"
dqn = DQN(env, alpha, gamma, epsilon,
          double=True)
dqn.Q_net.load_state_dict(torch.load(weights_dqn))
dqn_test = test_agent(env, dqn, n_episodes=10,
                      save=TEST+"/DDQN")


env.step = free_step(env)

twap = TWAP()
twap.train(env)
test = test_agent(env, twap, n_episodes=10,
                  save=TEST+"/twap")

pov = POV(0.1)
test = test_agent(env, pov, n_episodes=10, episode_retrain=True,
                  save=TEST+"/pov")
