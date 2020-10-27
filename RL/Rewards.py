def vwap_reward(env, action, state, done):
    price = env.orderbook['PRE_COMPRA1'][env.istep] * action
    mean = env.price_mean * action
    reward = mean - price
    return reward
