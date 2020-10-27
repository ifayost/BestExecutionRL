from RL.Environment import MarketGym
from RL.Rewards import vwap_reward
import numpy as np
import matplotlib.pyplot as plt


plt.style.use("solarized_dark")


PATH = "/Users/ifayost/Desktop/TFM/Projects/simplification/episodes"

env = MarketGym(PATH, vwap_reward, mode='train')

state = env.reset()
history = [state]
rewards = []
done = False
while not done:
    state, reward, done, info = env.step(0)  # env.action_space.sample())
    history.append(state)
    rewards.append(reward)
history = np.array(history)

print("Reward:", np.sum(rewards))

fig, ax = plt.subplots(len(env.variables) + 2, 1,
                       sharex=True, figsize=(12, 50), dpi=200)
prices = env.orderbook[env.priceCol][:len(env.history)]
ax[0].plot(range(len(env.history)), prices, label='Price')
ax[0].plot([0, len(env.history)],
           [env.price_mean, env.price_mean], linestyle='dashed', label='Mean')
ax[0].grid(True)
ax[0].legend()

ax[1].plot(range(history.shape[0]), history[:, 0], label='Time')
ax[1].plot(range(history.shape[0]), history[:, 1], label='Inventory')
ax[1].grid(True)
ax[1].legend()

for i, var in enumerate(env.variables):
    i += 2
    ax[i].plot(range(history.shape[0]), history[:, i], label=var)
    ax[i].grid(True)
    ax[i].legend(loc='lower left')
#plt.subplots_adjust(hspace=0.05)
plt.show()
