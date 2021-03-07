import pandas as pd
import numpy as np
import os
import pickle5
import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_df(path, test=None):
    files = [i for i in os.listdir(path)]
    files.remove('.DS_Store')
    dfs = {int(i.split('_')[-1][:-4]): pd.read_csv(os.path.join(path, i))
           for i in files}
    n_orders = {i: len(j) for i, j in dfs.items()}
    dfsc = dfs.copy()
    for i in dfsc.keys():
        if n_orders[i] == 0:
            del dfs[i]
    keys = sorted(list(dfs.keys()))
    if test is not None:
        train = {k: dfs[k] for k in keys[:-test]}
    else:
        train = {k: dfs[k] for k in keys}
    train = pd.concat(train.values(), ignore_index=True, sort='time')\
        .drop('FECHA', axis=1)
    train.time = pd.to_datetime(train.time)
    if test is not None:
        test = {k: dfs[k] for k in keys[-test:]}
        test = pd.concat(test.values(), ignore_index=True, sort='time')\
            .drop('FECHA', axis=1)
        test.time = pd.to_datetime(test.time)
        return train, test
    return train


def test_episode_agent(env, agent, name=None, random_seed=None,
                       episode_retrain=False):
    np.random.seed(random_seed)
    state = env.reset()

    hist = []
    actual_time = env.init_time
    time = []
    rewards = []
    done = False
    if episode_retrain:
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
    prices = np.array(env.orderbook[env.priceCol])

    # plt.rcParams['font.size'] = 18
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True,
                                             figsize=(20, 10), dpi=200)
    ax1.plot(time, prices[:n_steps], label='Price')
    ax1.plot(time, [env.price_mean] * len(time), linestyle='dashed',
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

    if save:
        directory = os.path.dirname(save)
        if not os.path.exists(directory):
            os.makedirs(directory)

    rewards = {}
    actions = []
    remaining_inventory = []
    prices = {}
    for episode in tqdm(range(n_episodes)):
        rewards[episode] = []
        prices[episode] = []
        state = env.reset()
        done = False
        if episode_retrain:
            agent.train(env)
        while not done:
            action = agent.predict(state)
            actions.append(action)
            prices[episode].append(env.orderbook[env.priceCol][env.istep])
            state, reward, done, info = env.step(action)
            rewards[episode].append(reward)
        if env.t <= pd.to_timedelta(0, unit='s'):
            remaining_inventory.append(action / env.V)

    if save is not None:
        with open(save + '.pickle', 'wb') as file:
            pickle5.dump([rewards, actions, remaining_inventory, prices],
                         file, protocol=pickle5.HIGHEST_PROTOCOL)

    return rewards, actions, remaining_inventory, prices


def open_test(path):
    with open(path, 'rb') as file:
        rewards, actions, remaining_inventory, cost = pickle5.load(file)
    return rewards, actions, remaining_inventory, cost


def round_value_error(x, e):
    x = round(x, -int(np.floor(np.log10(abs(x / 100)))))
    e = round(e, -int(np.floor(np.log10(abs(e / 100)))))
    return x, e


def plot_test(test_path, episodes):
    rewards, actions, remaining_inventory, _ = \
        open_test(test_path)
    cumulative_rewards = [sum(i) for i in rewards.values()]
    total_episodes = len(rewards.keys())
    with open(episodes + '/info.txt', 'r') as file:
        info = json.load(file)

    time_step = pd.to_timedelta(info['time_step'])
    n_steps_per_episode = [len(i) for i in rewards.values()]

    figures = 3
    figure_remaining_inventory = (any(np.array(remaining_inventory)) != 0)
    figures += figure_remaining_inventory
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

    if len(np.unique(actions)) > 10:
        ax[1].hist(actions, bins=20)
    else:
        c_actions = Counter(actions)
        keys = list(c_actions.keys())
        ax[1].bar(keys, c_actions.values())
    ax[1].set_title('Actions')
    ax[1].set_xlabel('Volume')

    hist = ax[2].hist(n_steps_per_episode, bins=20)
    xticks = ax[2].get_xticks()
    ax[2].set_xticklabels(pd.to_timedelta(xticks * time_step, unit='s')
                            .map(lambda x: str(x)[7:]), rotation=45)
    mean = np.mean(n_steps_per_episode)
    std = np.std(n_steps_per_episode)
    m = str(pd.to_timedelta(mean * time_step, unit='s'))[7:]
    s = str(pd.to_timedelta(std * time_step, unit='s'))[7:]
    ax[2].plot([mean, mean], [0, max(hist[0])*1.05], linewidth=3,
               c='#859900', label=f'Mean: {m} ± {s}')
    ax[2].set_title('Time to finish the episode')
    ax[2].set_xlabel('Time (H:M:S)')
    ax[2].legend()

    if figure_remaining_inventory:
        x = np.array(remaining_inventory)*100
        hist = ax[3].hist(x)
        mean = np.mean(x)
        std = np.std(x)
        if std != 0:
            m, s = round_value_error(mean, std)
        else:
            m, _ = round_value_error(mean, mean)
            s = std
        ax[3].plot([mean, mean], [0, max(hist[0])*1.05], linewidth=3,
                   c='#859900', label=f'Mean: {m:0.2e} ± {s:0.2e}')
        ax[3].set_title('Remaining inventory')
        ax[3].set_xlabel('Remaining inventory (%)')
        ax[3].legend()

    fig.suptitle(f'Tested on {total_episodes} episodes')


def plot_train_stats(stats, save=None, rolling=None):
    checks = np.array(stats['checkpoints'])
    rewards = np.array(stats['rewards'])
    epsilons = np.array(stats['epsilon'])
    if rolling is None:
        rolling = int(len(rewards[:, 0])/50)
    smooth = pd.DataFrame(rewards[:, 1]).rolling(rolling).mean()
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['font.size'] = 22
    fig, ax1 = plt.subplots()
    ax1.plot(rewards[:, 0], rewards[:, 1], alpha=0.5)
    ax1.plot(rewards[:, 0], smooth)
    ax1.scatter(checks[:, 0], checks[:, 1], c='#dc322f', marker='.')
    inf, sup = np.quantile(rewards[:, 1], [0.05, 0.95])
    ax1.set_ylim(inf, sup)
    ax1.set_ylabel('Reward')
    ax1.set_xlabel('Episode')
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = "#d33682"
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(rewards[:, 0], epsilons, color=color)
    ax2.fill_between(rewards[:, 0], epsilons,
                     interpolate=True, color=color, alpha=0.15)
    ax2.tick_params(axis='y', labelcolor=color)
    if save is not None:
        plt.savefig(save+'.png', dpi=200)
    plt.show()
