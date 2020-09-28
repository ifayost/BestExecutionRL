import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from collections import Counter
import json


def plot_state(env):
    history = [env.reset()]
    done = False
    while not done:
        state, reward, done, info = env.step(0)
        history.append(state)
    history = np.array(history)
    prices = env.orderbook[env.PriceCols[0]][:len(env.history)]
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20, 10), dpi=200)
    ax[0].plot(range(len(env.history)), prices, label='Price')
    ax[0].plot([0, len(env.history)], [env.avg, env.avg], linestyle='dashed',
               label='Epissode price average')
    ax[0].set_ylabel('Price')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(range(len(history)), history[:, 2], label='Standardized price')
    ax[1].plot(range(len(history)), history[:, 3], label='Standardized fma')
    ax[1].plot(range(len(history)), history[:, 4], label='Standardized sma')
    ax[1].legend()
    ax[1].grid(True)
    ax[2].plot(range(len(history)), history[:, 6], label='MACD')
    ax[2].plot(range(len(history)), history[:, 7], label='MACD signal')
    ax[2].bar(range(len(history)), history[:, 8], color='gold',
              label='MACD divergence')
    ax[2].legend()
    ax[2].grid(True)

    ax[3].plot(range(len(history)), history[:, 9], label='RSI')
    ax[3].legend()
    ax[3].grid(True)
    ax[3].set_ylabel('RSI')
    ax[3].set_xlabel('Step')
    plt.subplots_adjust(hspace=0.05)


def test_episode_agent(env, agent, name=None, random_seed=None, POV=False):
    np.random.seed(random_seed)
    state = env.reset()

    hist = []
    actual_time = env.init_time
    time = []
    rewards = []
    done = False
    if POV:
        agent.train(env)
    while not done:
        action = agent.predict(state)
        state, reward, done, _ = env.step(action)
        actual_time += env.time_step
        time.append(actual_time)
        hist.append(state)
        rewards.append(reward)
    hist = np.array(hist)

    n_steps = len(time)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True,
                                             figsize=(20, 10), dpi=200)
    for i in range(1):
        ax1.plot(time, env.orderbook['PRE_VENTA1'][:n_steps], c='green',
                 label='Price')
        # ax1.plot(time, env.orderbook['PRE_COMPRA1'][:n_steps], c='red')
    ax1.plot(time, [env.avg] * len(time), linestyle='dashed', c='#b58900',
             label='Episode price average')
    ax1.set_title('Orderbook')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time, hist[:, 1])
    ax2.set_title('Remaining Inventory')
    ax2.grid(True)

    ax3.plot(time, rewards)
    ax3.set_title('Rewards')
    ax3.set_ylabel('Rewards')
    ax3.grid(True)

    ax4.plot(time, env.history)
    ax4.set_title('Actions')
    ax4.set_ylabel('Volume')
    ax4.grid(True)
    ax4.set_xlabel('Time')

    plt.subplots_adjust(hspace=0.2)
    if name is not None:
        print(name)
    print("Total rewards:", f'{np.sum(rewards):,}')
    plt.show()


def test_agent(env, agent, n_episodes, save=None,
               random_seed=None, episode_retrain=False):
    np.random.seed(random_seed)

    rewards = {}
    actions = []
    remaining_inventory = []
    cost = {}
    for episode in tqdm(range(n_episodes)):
        rewards[episode] = []
        cost[episode] = []
        state = env.reset()
        done = False
        if episode_retrain:
            agent.train(env)
        while not done:
            action = agent.predict(state)
            actions.append(action)
            price = env.orderbook[env.PriceCols[0]][env.istep]
            state, reward, done, info = env.step(action)
            rewards[episode].append(reward)
            cost[episode].append(env.posible_actions[action] * price)
        if env.t <= pd.to_timedelta(0, unit='s'):
            remaining_inventory.append(action / env.V)

    if save is not None:
        with open(save + '.pickle', 'wb') as file:
            pickle.dump([rewards, actions, remaining_inventory, cost],
                        file, protocol=pickle.HIGHEST_PROTOCOL)

    return rewards, actions, remaining_inventory, cost


def open_test(path):
    with open(path, 'rb') as file:
        rewards, actions, remaining_inventory, cost = pickle.load(file)
    return rewards, actions, remaining_inventory, cost


def round_value_error(x, e):
    x = round(x, -int(np.floor(np.log10(abs(x / 100)))))
    e = round(e, -int(np.floor(np.log10(abs(e / 100)))))
    return x, e


def plot_test(path, episodes):
    posible_actions = np.array([0, 10, 100, 1_000, 10_000, 100_000])
    rewards, actions, remaining_inventory, cost = \
        open_test(path)
    cumulative_rewards = [sum(i) for i in rewards.values()]
    episode_cost = [sum(i) for i in cost.values()]
    total_episodes = len(rewards.keys())

    with open(episodes + '/info.txt', 'r') as file:
        info = json.load(file)

    inventory = int(info['volume'])
    time_step = pd.to_timedelta(info['time_step'])
    n_steps_per_episode = [len(i) for i in rewards.values()]

    figures = 4
    figure_remaining_inventory = (any(np.array(remaining_inventory)) != 0)
    figures += figure_remaining_inventory
    # figure_steps = (len(np.unique(n_steps_per_episode)) > 1)
    fig, ax = plt.subplots(1, figures)

    hist = ax[0].hist(cumulative_rewards, bins=20)
    mean = np.mean(cumulative_rewards)
    std = np.std(cumulative_rewards)
    m, s = round_value_error(mean, std)
    ax[0].plot([mean, mean], [0, max(hist[0])*1.05], linewidth=3, c='#859900',
               label=f'Mean: {m} ± {s}')
    ax[0].set_title('Metric')
    ax[0].set_xlabel('Reward')
    ax[0].legend()

    x = np.array(episode_cost)/inventory
    hist = ax[1].hist(x, bins=20)
    mean = np.mean(x)
    std = np.std(x)
    m, s = round_value_error(mean, std)
    ax[1].plot([mean, mean], [0, max(hist[0])*1.05], linewidth=3, c='#859900',
               label=f'Mean: {m} ± {s}')
    ax[1].set_title('Average cost per share')
    ax[1].set_xlabel('Price')
    ax[1].legend()

    c_actions = Counter(actions)
    x = range(len(posible_actions))
    y = np.zeros(len(posible_actions))
    for i, j in c_actions.items():
        y[i] = j/sum(c_actions.values())
    ax[2].bar(x, y)
    ax[2].set_xticks(np.arange(len(x)))
    ax[2].set_xticklabels(posible_actions, rotation=45)
    ax[2].set_title('Actions')
    ax[2].set_xlabel('Volume')

    hist = ax[3].hist(n_steps_per_episode, bins=20)
    xticks = ax[3].get_xticks()
    ax[3].set_xticklabels(pd.to_timedelta(xticks * time_step, unit='s')
                            .map(lambda x: str(x)[7:]), rotation=45)
    mean = np.mean(n_steps_per_episode)
    std = np.std(n_steps_per_episode)
    m = str(pd.to_timedelta(mean * time_step, unit='s'))[7:]
    s = str(pd.to_timedelta(std * time_step, unit='s'))[7:]
    ax[3].plot([mean, mean], [0, max(hist[0])*1.05], linewidth=3,
               c='#859900', label=f'Mean: {m} ± {s}')
    ax[3].set_title('Time to finish the episode')
    ax[3].set_xlabel('Time (H:M:S)')
    ax[3].legend()

    if figure_remaining_inventory:
        x = np.array(remaining_inventory)*100
        hist = ax[4].hist(x)
        mean = np.mean(x)
        std = np.std(x)
        if std != 0:
            m, s = round_value_error(mean, std)
        else:
            m, _ = round_value_error(mean, mean)
            s = std
        ax[4].plot([mean, mean], [0, max(hist[0])*1.05], linewidth=3,
                   c='#859900', label=f'Mean: {m:0.2e} ± {s:0.2e}')
        ax[4].set_title('Remaining inventory')
        ax[4].set_xlabel('Remaining inventory (%)')
        ax[4].legend()

    fig.suptitle(f'Tested on {total_episodes} episodes')
