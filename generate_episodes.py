from RL.Environment import EpisodeGenerator, make_variables
from RL.Utils import read_df
import pandas as pd


PATH = "/Users/ifayost/Desktop/TFM/Projects"\
    "/DATA/SAN_month/best_exec_data/orderbook/"
SAVE = "/Users/ifayost/Desktop/TFM/Projects/simplification/episodes"

train, test = read_df(PATH, test=5)

H = pd.to_timedelta(1, unit='hour')
V = 500_000
buy = True
time_step = pd.to_timedelta(1, unit='s')
make_variables = make_variables
eg = EpisodeGenerator(H, V, buy, time_step, make_variables)
vardict = eg.generate_episodes(SAVE, 1000, train, test)
