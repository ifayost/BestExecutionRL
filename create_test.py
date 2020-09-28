import torch
import matplotlib.pyplot as plt
from RL.Environments import MarketGym
from RL.Agents import TWAP, POV, DQN, dynamicTWAP
from RL.Rewards import mean_step
from RL.Utils import test_agent

plt.style.use("solarized_dark")

EPISODES = "./episodes"
TEST = "./test"

env = MarketGym(EPISODES, mean_step)


alpha = 5e-4
gamma = 0.999
epsilon = 0.1

weights_dqn = "./weights/DDQN_mean-end1e6.pt"
dqn = DQN(env, alpha, gamma, epsilon,
          double=True, save=weights_dqn)
dqn.Q_net.load_state_dict(torch.load(weights_dqn))

dqn_test = test_agent(env, dqn, len(env.episodes),
                      save=TEST+"/DDQN_mean-end1e6")


weights_dqn = "./weights/DDQN_mean.pt"
dqn = DQN(env, alpha, gamma, epsilon,
          double=True, save=weights_dqn)
dqn.Q_net.load_state_dict(torch.load(weights_dqn))

dqn_test = test_agent(env, dqn, len(env.episodes),
                      save=TEST+"/DDQN_mean")


weights_dqn = "./weights/DDQN_mean+1step-end1e6.pt"
dqn = DQN(env, alpha, gamma, epsilon,
          double=True, save=weights_dqn)
dqn.Q_net.load_state_dict(torch.load(weights_dqn))

dqn_test = test_agent(env, dqn, len(env.episodes),
                      save=TEST+"/DDQN_mean+1step-end1e6")


weights_dqn = "./weights/DDQN_mean_step.pt"
dqn = DQN(env, alpha, gamma, epsilon,
          double=True, save=weights_dqn)
dqn.Q_net.load_state_dict(torch.load(weights_dqn))

dqn_test = test_agent(env, dqn, len(env.episodes),
                      save=TEST+"/DDQN_mean_step")


twap = TWAP()
twap.train(env)
test = test_agent(env, twap, len(env.episodes),
                  save=TEST+"/twap")

dtwap = dynamicTWAP()
dtwap.train(env)
test = test_agent(env, dtwap, len(env.episodes), episode_retrain=True,
                  save=TEST+"/dtwap")

pov = POV(0.1)
test = test_agent(env, pov, len(env.episodes), episode_retrain=True,
                  save=TEST+"/pov")
