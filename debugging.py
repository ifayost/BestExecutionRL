import numpy as np
import matplotlib.pyplot as plt
from RL.Environments import MarketGym
from RL.Rewards import mean_episodic

plt.style.use("solarized_dark")

PATH = "./episodes"

env = MarketGym(PATH, mean_episodic)

history = [env.reset()]
state = env.reset()
done = False
while not done:
    state, reward, done, info = env.step(0)  # env.action_space.sample())
    history.append(state)
history = np.array(history)

print("Reward:", reward)

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 14), dpi=200)
prices = env.orderbook[env.PriceCols[0]][:len(env.history)]
ax[0].plot(range(len(env.history)), prices)
ax[0].plot([0, len(env.history)], [env.avg, env.avg], linestyle='dashed')
ax[0].grid(True)
ax[1].plot(range(len(history)), history[:, 2])
ax[1].plot(range(len(history)), history[:, 3])
ax[1].plot(range(len(history)), history[:, 4])
ax[1].grid(True)
ax[2].plot(range(len(history)), history[:, 6])
ax[2].plot(range(len(history)), history[:, 7])
ax[2].bar(range(len(history)), history[:, 8], color='gold')
ax[2].grid(True)
ax[3].plot(range(len(history)), history[:, 9])
ax[3].grid(True)
plt.subplots_adjust(hspace=0.05)
plt.show()
