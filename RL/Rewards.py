import numpy as np
import pandas as pd


def mean_step(env, action, state, done):
    reward = env.orderbook[env.PriceCols[0]][env.istep] * action
    mean = env.avg * action
    reward = mean - reward
    reward = reward if env.buy else -reward
    # if env.t <= pd.to_timedelta(0, unit='s'):
    #     reward = reward - 100_000 * (1 + action / env.V)
    return reward


def mean_episodic(env, action, state, done):
    if not done:
        reward = 0
    else:
        mean = env.avg * env.V
        n_steps = len(env.history)
        reward = np.sum(np.array(env.orderbook[env.PriceCols[0]][:n_steps]) *
                        np.array(env.history))
        reward = mean - reward
        reward = reward if env.buy else -reward
        if env.t <= pd.to_timedelta(0, unit='s'):
            reward = reward - 1_000_000 * (1 + action / env.V)
    return reward


def simple_mean(env, action, state, done):
    if not done:
        reward = 0
    else:
        mean = env.avg * env.V
        n_steps = len(env.history)
        reward = np.sum(np.array(env.orderbook[env.PriceCols[0]][:n_steps]) *
                        np.array(env.history))
        reward = mean - reward
        reward = reward if env.buy else -reward
    return reward
