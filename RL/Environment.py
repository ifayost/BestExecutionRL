import gym
from gym import spaces
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import json
import os


class MarketGym(gym.Env):
    """Market Gym"""
    def __init__(self, path, reward_function, mode='train'):
        super(MarketGym).__init__()
        self.path = path
        self.compute_reward = reward_function
        self.mode = mode

        # Episode reading:
        try:
            info = self.read_info(path)
            self.H = pd.to_timedelta(info['horizon'])
            self.t = self.H
            self.V = np.abs(int(info['volume']))
            self.i = self.V
            self.buy = bool(info['buy'])
            self.priceCol = 'PRE_VENTA1' if self.buy else 'PRE_COMPRA1'
            self.volumeCol = \
                'VORDEN_PRVENTA1' if self.buy else 'VORDEN_PRCOMPRA1'
            self.volMeanCol = \
                'VORDEN_PRCOMPRA1' if self.buy else 'VORDEN_PRVENTA1'
            self.time_step = pd.to_timedelta(info['time_step'])
            self.keys = info['variables']
            self.episodes = []
            for i in os.listdir(os.path.join(path, mode)):
                if i[-7:] == '.pickle':
                    self.episodes.append(i)
            self.expected_steps = self.H.delta / self.time_step.delta
            self.history = []
        except FileNotFoundError:
            print('There are not episodes compatibles with the environment '
                  f'in {path}. Try with another path or create some episodes.')

        # Variable initialization:
        self.orderbook = None
        self.done = False
        self.istep = 0
        self.state = []
        self.volume_mean = 0
        self.price_mean = 0

        # Action space definition:
        self.action_space = spaces.Discrete(2)
        self.ord_size = 0.2 * self.volume_mean
        self.posible_actions = [0, self.ord_size]

        # Observation space definition:
        ignore = ['time', 'VORDEN_PRCOMPRA1', 'PRE_COMPRA1', 'PRE_VENTA1',
                  'VORDEN_PRVENTA1', 'FECHA', 'vol_acc']
        self.variables = [k for k in self.keys if k not in ignore]
        self.variables = [self.variables[0]] + self.variables[3:]
        self.observation_space = \
            spaces.Box(low=np.array([0, 0] + [-5] * len(self.variables)),
                       high=np.array([1, 1] + [5] * len(self.variables)),
                       dtype=np.float32)

    def read_info(self, path):
        with open(path + '/info.txt', 'r') as file:
            info = json.load(file)
        return info

    def make_state(self, istep):
        self.volume_mean += \
            (self.orderbook[self.volMeanCol][istep] - self.volume_mean) / \
            (istep + 1)
        self.ord_size = 0.2 * self.volume_mean
        self.posible_actions = [0, self.ord_size]

        self.price_mean += \
            (self.orderbook[self.priceCol][istep] - self.price_mean) / \
            (istep + 1)

        state = [self.t.delta/self.H.delta, self.i/self.V]
        state = state + [self.orderbook[col][istep] for col in self.variables]
        state = np.array(state, dtype='float32')
        return state

    def step(self, action):
        action = self.posible_actions[action]
        self.history.append(action)

        self.t -= self.time_step

        # Comprobamos que no se ha terminado el tiempo o el inventario.
        # Si se ha terminado el tiemo vendemos/compramos todo lo que queda.
        if self.t <= pd.to_timedelta(0, unit='s') or action >= self.i:
            action = self.i
            self.i = 0
            self.done = True
        else:
            self.i -= action

        reward = self.compute_reward(self, action, self.step, self.done)

        self.istep += 1
        self.state = self.make_state(self.istep)

        info = {}

        return self.state, reward, self.done, info

    def reset(self):
        self.history = []
        episode_number = np.random.choice(self.episodes)
        with open(os.path.join(self.path, self.mode, episode_number),
                  'rb') as file:
            self.orderbook = pickle.load(file)
        self.init_time = pd.to_datetime(self.orderbook['time'][0])
        self.volume_mean = 0
        self.price_mean = 0
        self.istep = 0
        self.t = self.H
        self.i = self.V
        self.done = False
        self.state = self.make_state(self.istep)
        return self.state


class EpisodeGenerator:
    def __init__(self, horizon, volume, buy, time_step, make_variables):
        self.H = horizon
        self.V = volume
        self.buy = buy
        self.time_step = time_step
        self.cols = ['PRE_COMPRA1', 'VORDEN_PRCOMPRA1',
                     'PRE_VENTA1', 'VORDEN_PRVENTA1']
        self.make_variables = make_variables

    def generate_episodes(self, path, n_episodes, train, test=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, 'train')):
            os.makedirs(os.path.join(path, 'train'))
        if not os.path.exists(os.path.join(path, 'test')):
            os.makedirs(os.path.join(path, 'test'))

        for test_bool, df in enumerate([train, test]):
            # Posiciones del orderbook que permiten completar
            # el tiempo entero del horizonte.
            start_positions = np.array(
                df.time[df.time.dt.time <=
                        (pd.to_datetime("17:35:00.000000") -
                         self.H).time()]
                .index)
            if test_bool:
                n_episodes = int(n_episodes * (len(test)/len(train)))
            start_positions = np.random.choice(start_positions,
                                               size=n_episodes,
                                               replace=False)

            for pos in tqdm(start_positions):
                episode = {k: [] for k in self.cols}
                episode['time'] = []
                time = df.iloc[pos].time
                fin = time + self.H
                while time <= fin:
                    episode['time'].append(time)
                    state = df.iloc[pos]
                    for col in self.cols:
                        episode[col].append(state[col])
                    time += self.time_step
                    pos = df.index[df.time <= time][-1]
                episode = self.make_variables(self, episode)
                with open(os.path.join(path, ['train', 'test'][test_bool],
                                       str(pos)) + '.pickle', 'wb') as file:
                    pickle.dump(episode, file,
                                protocol=pickle.HIGHEST_PROTOCOL)

        info = {'horizon': str(self.H), 'volume': str(self.V),
                'buy': str(self.buy), 'time_step': str(self.time_step),
                'variables': list(episode.keys())}

        with open(os.path.join(path, 'info.txt'), 'w') as file:
            json.dump(info, file)


def free_step(env):

    def step(action):
        env.history.append(action)

        env.t -= env.time_step

        # Comprobamos que no se ha terminado el tiempo o el inventario.
        # Si se ha terminado el tiemo vendemos/compramos todo lo que queda.
        if env.t <= pd.to_timedelta(0, unit='s') or action >= env.i:
            action = env.i
            env.i = 0
            env.done = True
        else:
            env.i -= action

        reward = env.compute_reward(env, action, env.step, env.done)

        env.istep += 1
        env.state = env.make_state(env.istep)

        info = {}

        return env.state, reward, env.done, info
    return step


def make_variables(env, df, windows=[4, 6, 18, 30], clip_min=-5, clip_max=5):
    orderbook = df.copy()
    windows = sorted(windows)
    expected_steps = env.H.delta / env.time_step.delta
    windows = [int(expected_steps/w) for w in windows]

    pc = 'PRE_VENTA1' if env.buy else 'PRE_COMPRA1'
    vc = 'VORDEN_PRVENTA1' if env.buy else 'VORDEN_PRCOMPRA1'
    prices = pd.Series(orderbook[pc])
    volume = pd.Series(orderbook[vc])

    def standarize(x, windows, clip_min, clip_max):
        x = pd.Series(x.copy())
        avg_ = np.nan_to_num(
            x.rolling(windows[0], min_periods=0).mean(), nan=0)
        std_ = np.nan_to_num(
            x.rolling(windows[0], min_periods=0).std(), nan=1)
        standarized = np.clip(
            np.nan_to_num((x - avg_)/std_, 0), clip_min, clip_max)
        return standarized

    # Base avg and std prices
    avg = \
        np.nan_to_num(prices.rolling(windows[0], min_periods=0).mean(), nan=0)
    std = \
        np.nan_to_num(prices.rolling(windows[0], min_periods=0).std(), nan=1)
    orderbook['price'] = np.clip((prices - avg)/std, clip_min, clip_max)
    orderbook[f'{windows[0]}ma'] = avg
    orderbook[f'{windows[0]}std'] = std

    # Price MAs
    for w in windows[1:]:
        orderbook[f'{w}ma'] = np.nan_to_num(
            (prices.rolling(w).mean() - avg) / std, 0)

    # Volatility
    for w in windows[1:]:
        orderbook[f'{w}std'] = \
            np.nan_to_num(prices.rolling(w, min_periods=0).std(), nan=1)
        orderbook[f'{w}std'] = \
            np.clip(np.nan_to_num(orderbook[f'{w}std']/std, nan=0),
                    clip_min, clip_max)

    # Macd (emparejamos ventanas rapidas con ventanas lentas)
    w_fast = windows[int(len(windows)/2) + 1:][::-1]
    w_slow = windows[1:int((len(windows)-1)/2) + 1]
    mid = windows[int(len(windows)/2)]
    for f, s in zip(w_fast, w_slow):
        macd = pd.Series(orderbook[f'{f}ma'] - orderbook[f'{s}ma'])\
            .clip(clip_min, clip_max)
        orderbook[f'{f}_{s}_macd'] = macd.to_numpy()
        orderbook[f'{f}_{s}_macd_signal'] = macd.ewm(halflife=mid).mean()
        orderbook[f'{f}_{s}_macd_div'] = \
            orderbook[f'{f}_{s}_macd'] - orderbook[f'{f}_{s}_macd_signal']

    # RSIs
    def make_rsi(prices, w_period):
        dif = prices.diff(1)
        up = dif.copy()
        down = dif.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        up = up.rolling(w_period).mean()
        down = abs(down.rolling(w_period).mean())
        rsi = 1 - 1 / (1 + (up/down))
        return rsi.values

    for w in windows[1:]:
        rsi = make_rsi(prices, w)
        orderbook[f'{w}rsi'] = np.nan_to_num(rsi - 0.5)

    # Accumulated Volume
    orderbook['vol_acc'] = volume.cumsum()

    # MFIs
    pricexvolume = prices * volume
    for w in windows[1:]:
        mfi = make_rsi(pricexvolume, w)
        orderbook[f'{w}mfi'] = \
            standarize(mfi, windows, clip_min, clip_max)

    # Volumes
    volumes = []
    for w in windows:
        volumes.append(volume.rolling(w, min_periods=0))

    # VWAPs
    orderbook['vwap_acc'] = pricexvolume.cumsum() / orderbook['vol_acc']
    log_vwap_acc = np.log(orderbook['vwap_acc'])

    for v, w in zip(volumes, windows):
        orderbook[f'{w}vwap'] = np.nan_to_num(
            pricexvolume.rolling(w, min_periods=0).sum() / v.sum(),
            nan=0)

        vwap_log_vwap_acc = orderbook[f'{w}vwap'] / log_vwap_acc
        orderbook[f'{w}vwap/log(vwap_acc)'] = \
            standarize(vwap_log_vwap_acc,
                       windows, clip_min, clip_max)
        orderbook[f'{w}vwap'] = \
            standarize(orderbook[f'{w}vwap'], windows, clip_min, clip_max)

    orderbook['vwap_acc'] = \
        standarize(orderbook['vwap_acc'], windows, clip_min, clip_max)

    # Price/Volume
    for w in windows:
        orderbook[f'{w}p/v'] = np.log(prices/volume)
        orderbook[f'{w}p/v'] = \
            standarize(orderbook[f'{w}p/v'], windows, clip_min, clip_max)

    # Spread
    orderbook['spread'] = \
        pd.Series(orderbook['PRE_VENTA1']) - \
        pd.Series(orderbook['PRE_COMPRA1'])
    for w in windows[1:]:
        orderbook[f'{w}spread'] = np.nan_to_num(
            orderbook['spread'].rolling(w, min_periods=0).mean(), nan=0)
        orderbook[f'{w}spread'] = \
            standarize(orderbook[f'{w}spread'], windows, clip_min, clip_max)
    orderbook['spread'] = \
        standarize(orderbook['spread'], windows, clip_min, clip_max)

    return orderbook


def make_variables_simple(env, df, windows=[4, 10, 30], clip_min=-5,
                          clip_max=5):
    orderbook = df.copy()
    windows = sorted(windows)
    expected_steps = env.H.delta / env.time_step.delta
    windows = [int(expected_steps/w) for w in windows]

    pc = 'PRE_VENTA1' if env.buy else 'PRE_COMPRA1'
    vc = 'VORDEN_PRVENTA1' if env.buy else 'VORDEN_PRCOMPRA1'
    prices = pd.Series(orderbook[pc])
    volume = pd.Series(orderbook[vc])

    def standarize(x, windows, clip_min, clip_max):
        x = pd.Series(x.copy())
        avg_ = np.nan_to_num(
            x.rolling(windows[0], min_periods=0).mean(), nan=0)
        std_ = np.nan_to_num(
            x.rolling(windows[0], min_periods=0).std(), nan=1)
        standarized = np.clip(
            np.nan_to_num((x - avg_)/std_, 0), clip_min, clip_max)
        return standarized

    # Base avg and std prices
    avg = \
        np.nan_to_num(prices.rolling(windows[0], min_periods=0).mean(), nan=0)
    std = \
        np.nan_to_num(prices.rolling(windows[0], min_periods=0).std(), nan=1)
    orderbook['price'] = np.clip((prices - avg)/std, clip_min, clip_max)
    orderbook[f'{windows[0]}ma'] = avg
    orderbook[f'{windows[0]}std'] = std

    # Price MAs
    for w in windows[1:]:
        orderbook[f'{w}ma'] = np.nan_to_num(
            (prices.rolling(w).mean() - avg) / std, 0)

    # RSIs
    def make_rsi(prices, w_period):
        dif = prices.diff(1)
        up = dif.copy()
        down = dif.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        up = up.rolling(w_period).mean()
        down = abs(down.rolling(w_period).mean())
        rsi = 1 - 1 / (1 + (up/down))
        return rsi.values

    rsi = make_rsi(prices, windows[1])
    orderbook[f'{windows[1]}rsi'] = np.nan_to_num(rsi - 0.5)

    # MFIs
    pricexvolume = prices * volume
    mfi = make_rsi(pricexvolume, windows[1])
    orderbook[f'{windows[1]}mfi'] = \
        standarize(mfi, windows, clip_min, clip_max)

    # Price/Volume
    orderbook['p/v'] = \
        standarize(np.log(prices/volume), windows, clip_min, clip_max)

    # Spread
    orderbook['spread'] = \
        pd.Series(orderbook['PRE_VENTA1']) - \
        pd.Series(orderbook['PRE_COMPRA1'])
    orderbook['spread'] = \
        standarize(orderbook['spread'], windows, clip_min, clip_max)

    return orderbook
