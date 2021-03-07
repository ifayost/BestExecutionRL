import pandas as pd


def vwap_reward(env, action, state, done):
    price = env.orderbook['PRE_COMPRA1'][env.istep] * action
    mean = env.price_mean * action
    reward = mean - price
    return reward


def vwap_reward_penalty(penalty):
    def reward_function(env, action, state, done):
        price = env.orderbook['PRE_COMPRA1'][env.istep] * action
        mean = env.price_mean * action
        reward = mean - price
        if env.t <= pd.to_timedelta(0, unit='s'):
            reward = -10000 * abs(reward)
        return reward
    return reward_function
