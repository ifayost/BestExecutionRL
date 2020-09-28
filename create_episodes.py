import pandas as pd
from RL.Environments import MarketGym

N_EPISODES = 2000
PATH = "./episodes"
ORDERBOOK_PATH = "../DATA/SAN/san_orderbooks.csv"

HORIZON = pd.to_timedelta(20, unit='min')
INVENTORY = 500_000
TIME_DELTA = pd.to_timedelta(1, unit='s')

env = MarketGym(PATH, None)
episode_maker = env.episode_maker(HORIZON, INVENTORY, TIME_DELTA)
episode_maker.read_orderbook(ORDERBOOK_PATH, n_levels=1)
episode_maker.make_episodes(env.path, N_EPISODES)
