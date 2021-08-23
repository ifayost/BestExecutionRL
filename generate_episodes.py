from RL.Environment import EpisodeGenerator, make_variables_simple
from RL.Utils import read_df
import pandas as pd


PATH = "../DATA/"
SAVE = "./episodes/simple"

train, test = read_df(PATH, test=5)

H = pd.to_timedelta(1, unit='hour')
V = 500_000
buy = True
time_step = pd.to_timedelta(1, unit='s')
make_variables = make_variables_simple
eg = EpisodeGenerator(H, V, buy, time_step, make_variables)
vardict = eg.generate_episodes(SAVE, 1000, train, test)
