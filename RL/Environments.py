import gym
from gym import spaces
import pandas as pd
import numpy as np
import pickle
import json
import os
from tqdm import tqdm


class MarketGym(gym.Env):
    """Market Gym"""
    def __init__(self, path, reward_function):
        super(MarketGym).__init__()
        self.path = path
        self.compute_reward = reward_function
        try:
            info = self.read_info(path)
            self.H = pd.to_timedelta(info['horizon'])
            self.t = self.H
            self.V = np.abs(int(info['volume']))
            self.i = self.V
            self.buy = bool(info['buy'])
            self.PriceCols = info['PriceCols']
            self.SizeCols = info['SizeCols']
            self.time_step = pd.to_timedelta(info['time_step'])
            self.episodes = []
            for i in os.listdir(path):
                if i[-7:] == '.pickle':
                    self.episodes.append(i)

            self.avg = 0
            self.expected_steps = self.H.delta / self.time_step.delta
            self.fast = int(self.expected_steps / 30)
            self.mid = int(self.expected_steps / 18)
            self.slow = int(self.expected_steps / 6)
            self.base = int(self.expected_steps / 4)
            self.history = []
        except FileNotFoundError:
            print('There are not episodes compatibles with the environment '
                  f'in {path}. Try with another path or create some episodes.')

        self.orderbook = None
        self.done = False
        self.istep = 0
        self.state = []
        self.action_space = spaces.Discrete(6)
        self.posible_actions = np.array([0, 10, 100, 1_000, 10_000, 100_000])
        self.observation_space = spaces.Box(low=np.array([0, 0, -4, -4, -4,
                                                          0, -4, -4, -4, 0]),
                                            high=np.array([1, 1, 4, 4, 4,
                                                           'inf', 4, 4, 4, 1]),
                                            dtype=np.float32)

    def read_info(self, path):
        with open(path + '/info.txt', 'r') as file:
            info = json.load(file)
        return info

    def make_state(self, istep):
        price = self.orderbook[self.PriceCols[0]][istep]
        avg = self.orderbook['bma'][istep]
        std = self.orderbook['smstd'][istep]
        state = \
            np.array([self.t.delta/self.H.delta,
                      self.i/self.V,
                      np.clip((price - avg)/std, -4, 4),
                      np.clip(self.orderbook['fma'][istep], -4, 4),
                      np.clip(self.orderbook['sma'][istep], -4, 4),
                      std,
                      self.orderbook['macd'][istep],
                      np.clip(self.orderbook['macd_signal'][istep],
                              -4, 4),
                      np.clip(self.orderbook['macd_div'][istep],
                              -4, 4),
                      self.orderbook['rsi'][istep]])\
            .astype(np.float32)
        return np.nan_to_num(state)

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
        with open(self.path + '/' + episode_number, 'rb') as file:
            self.orderbook = pickle.load(file)
        self.init_time = pd.to_datetime(self.orderbook['time'])
        self.avg = np.mean(self.orderbook[self.PriceCols[0]])
        self.istep = 0
        self.t = self.H
        self.i = self.V
        self.done = False
        self.state = self.make_state(self.istep)
        return self.state

    class episode_maker:
        def __init__(self, horizon, volume,
                     time_step=pd.to_timedelta(0.5, unit='s')):
            self.H = horizon
            self.t = horizon
            self.V = np.abs(volume)
            self.i = self.V
            self.buy = True if volume > 0 else False
            self.time_step = time_step
            self.expected_steps = self.H.delta / self.time_step.delta
            self.fast = int(self.expected_steps / 30)
            self.mid = int(self.expected_steps / 18)
            self.slow = int(self.expected_steps / 6)
            self.base = int(self.expected_steps / 4)

        def read_orderbook(self, path, n_levels):
            colnames = ['FECHA', 'HORA', 'CENTSEG']
            for i in range(1, n_levels+1):
                colnames.extend(['PRE_VENTA' + str(i),
                                 'VORDEN_PRVENTA' + str(i),
                                 'PRE_COMPRA' + str(i),
                                 'VORDEN_PRCOMPRA' + str(i)])
            orderbook = pd.read_csv(path, sep=';', engine='python')[colnames]
            orderbook['TIME'] = pd.to_datetime(orderbook['FECHA'] + ' ' +
                                               orderbook['HORA'] + ' ' +
                                               (orderbook['CENTSEG']*1000)
                                               .astype(str),
                                               format='%Y-%m-%d %H:%M:%S %f')
            orderbook = orderbook.drop(['FECHA', 'HORA', 'CENTSEG'], axis=1)
            orderbook = orderbook.sort_values(by='TIME')
            self.orderbook = orderbook.reset_index(drop=True)
            self.AskPriceCols = ['PRE_VENTA'+str(i)
                                 for i in range(1, n_levels+1)]
            self.AskSizeCols = ['VORDEN_PRVENTA'+str(i)
                                for i in range(1, n_levels+1)]
            self.BidPriceCols = ['PRE_COMPRA'+str(i)
                                 for i in range(1, n_levels+1)]
            self.BidSizeCols = ['VORDEN_PRCOMPRA'+str(i)
                                for i in range(1, n_levels+1)]

            # Posiciones del orderbook que permiten completar
            # el tiempo entero del horizonte.
            self.start_positions = np.array(
                orderbook.TIME[orderbook.TIME.dt.time <=
                               (pd.to_datetime("17:35:00.000000") -
                                self.H).time()]
                .index)

        def make_variables(self, orderbook):
            pc = self.PriceCols
            prices = pd.Series(orderbook[pc[0]])
            avg = \
                np.nan_to_num(prices.rolling(self.base,
                                             min_periods=0).mean(), nan=0)
            orderbook['bma'] = avg
            std = \
                np.nan_to_num(prices.rolling(self.base,
                                             min_periods=0).std(), nan=1)
            orderbook['smstd'] = std
            orderbook['fma'] = np.nan_to_num(
                (prices.rolling(self.fast).mean() - avg) / std)
            orderbook['sma'] = np.nan_to_num(
                (prices.rolling(self.slow).mean() - avg) / std)
            macd = pd.Series(orderbook['fma'] - orderbook['sma']).clip(-4, 4)
            orderbook['macd'] = macd.to_numpy()
            orderbook['macd_signal'] = macd.ewm(halflife=self.mid).mean()
            orderbook['macd_div'] = \
                orderbook['macd'] - orderbook['macd_signal']

            def make_rsi(prices, w_period):
                dif = prices.diff(1)
                up = dif.copy()
                down = dif.copy()
                up[up < 0] = 0
                down[down > 0] = 0
                up = up.rolling(w_period).mean()
                down = abs(down.rolling(w_period).mean())
                rs = up / down
                rsi = 1 - 1 / (1 + rs)
                return rsi.values
            orderbook['rsi'] = np.nan_to_num(make_rsi(prices, self.mid))

            return orderbook

        def make_episodes(self, path, n_episodes):

            if not os.path.exists(path):
                os.makedirs(path)

            if self.buy:
                self.PriceCols = self.AskPriceCols
                self.SizeCols = self.AskSizeCols
            else:
                self.PriceCols = self.BidPriceCols
                self.SizeCols = self.BidSizeCols

            info = {'horizon': str(self.H), 'volume': str(self.V),
                    'buy': str(self.buy), 'time_step': str(self.time_step),
                    'n_episodes': str(n_episodes), 'PriceCols': self.PriceCols,
                    'SizeCols': self.SizeCols}
            with open(path + '/info.txt', 'w') as file:
                json.dump(info, file)

            orderbook = self.orderbook[self.PriceCols + self.SizeCols +
                                       ['TIME']]

            start_positions = np.random.choice(self.start_positions,
                                               size=n_episodes,
                                               replace=False)
            for pos in tqdm(start_positions):
                episode = {key: [] for key in self.PriceCols + self.SizeCols}
                time = orderbook.iloc[pos].TIME
                fin = time + self.H
                episode['time'] = [time]
                while time <= fin:
                    state = orderbook.iloc[pos]
                    for pc in self.PriceCols:
                        episode[pc].append(state[pc])
                    for sc in self.SizeCols:
                        episode[sc].append(state[sc])
                    time += self.time_step
                    pos = orderbook.index[orderbook.TIME <= time][-1]
                episode = self.make_variables(episode)
                with open(path + '/' + str(pos) + '.pickle', 'wb') as file:
                    pickle.dump(episode, file,
                                protocol=pickle.HIGHEST_PROTOCOL)
