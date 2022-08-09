from lib import *


def read_data(date, instrument, time_step):
    path = os.path.join(PRICE_FLD, date, instrument + '.csv')
    if not os.path.exists(path):
        print('no such file: ' + path)
        return None

    df_raw = pd.read_csv(path, parse_dates=['time'], index_col='time')
    df = df_raw.resample(time_step, how='last').fillna(method='ffill')
    return df['spot'].values


class Sampler:

    def load_db(self, fld):

        self.db = pickle.load(open(os.path.join(fld, 'db.pickle'), 'rb'))
        param = json.load(open(os.path.join(fld, 'param.json'), 'rb'))
        self.i_db = 0
        self.n_db = param['n_episodes']
        self.sample = self.__sample_db
        for attr in param:
            if hasattr(self, attr):
                setattr(self, attr, param[attr])
        self.title = 'DB_' + param['title']

    def build_db(self, n_episodes, fld):
        db = []
        for i in range(n_episodes):
            prices, title = self.sample()
            db.append((prices, '[%i]_' % i + title))
        os.makedirs(fld)  # don't overwrite existing fld
        pickle.dump(db, open(os.path.join(fld, 'db.pickle'), 'wb'))
        param = {'n_episodes': n_episodes}
        for k in self.attrs:
            param[k] = getattr(self, k)
        json.dump(param, open(os.path.join(fld, 'param.json'), 'w'))

    def __sample_db(self):
        prices, title = self.db[self.i_db]
        self.i_db += 1
        if self.i_db == self.n_db:
            self.i_db = 0
        return prices, title


class SinSampler(Sampler):

    def __init__(self, game,
                 window_episode=None, noise_amplitude_ratio=None, period_range=None, amplitude_range=None,
                 fld=None):

        self.n_var = 1  # price only

        self.window_episode = window_episode
        self.noise_amplitude_ratio = noise_amplitude_ratio
        self.period_range = period_range
        self.amplitude_range = amplitude_range
        self.can_half_period = False

        self.attrs = ['title', 'window_episode', 'noise_amplitude_ratio', 'period_range', 'amplitude_range',
                      'can_half_period']

        param_str = str((
            self.noise_amplitude_ratio, self.period_range, self.amplitude_range
        ))
        if game == 'single':
            self.sample = self.__sample_single_sin
            self.title = 'SingleSin' + param_str
        elif game == 'concat':
            self.sample = self.__sample_concat_sin
            self.title = 'ConcatSin' + param_str
        elif game == 'concat_half':
            self.can_half_period = True
            self.sample = self.__sample_concat_sin
            self.title = 'ConcatHalfSin' + param_str
        elif game == 'concat_half_base':
            self.can_half_period = True
            self.sample = self.__sample_concat_sin_w_base
            self.title = 'ConcatHalfSin+Base' + param_str
            self.base_period_range = (int(2 * self.period_range[1]), 4 * self.period_range[1])
            self.base_amplitude_range = (20, 80)
        elif game == 'load':
            self.load_db(fld)
        else:
            raise ValueError

    def __rand_sin(self,
                   period_range=None, amplitude_range=None, noise_amplitude_ratio=None, full_episode=False):

        if period_range is None:
            period_range = self.period_range
        if amplitude_range is None:
            amplitude_range = self.amplitude_range
        if noise_amplitude_ratio is None:
            noise_amplitude_ratio = self.noise_amplitude_ratio

        period = random.randrange(period_range[0], period_range[1])
        amplitude = random.randrange(amplitude_range[0], amplitude_range[1])
        noise = noise_amplitude_ratio * amplitude

        if full_episode:
            length = self.window_episode
        else:
            if self.can_half_period:
                length = int(random.randrange(1, 4) * 0.5 * period)
            else:
                length = period

        p = 100. + amplitude * np.sin(np.array(range(length)) * 2 * 3.1416 / period)
        p += np.random.random(p.shape) * noise

        return p, '100+%isin((2pi/%i)t)+%ie' % (amplitude, period, noise)

    def __sample_concat_sin(self):
        prices = []
        p = []
        while True:
            p = np.append(p, self.__rand_sin(full_episode=False)[0])
            if len(p) > self.window_episode:
                break
        prices.append(p[:self.window_episode])
        return np.array(prices).T, 'concat sin'

    def __sample_concat_sin_w_base(self):
        prices = []
        p = []
        while True:
            p = np.append(p, self.__rand_sin(full_episode=False)[0])
            if len(p) > self.window_episode:
                break
        base, base_title = self.__rand_sin(
            period_range=self.base_period_range,
            amplitude_range=self.base_amplitude_range,
            noise_amplitude_ratio=0.,
            full_episode=True)
        prices.append(p[:self.window_episode] + base)
        return np.array(prices).T, 'concat sin + base: ' + base_title

    def __sample_single_sin(self):
        prices = []
        funcs = []
        p, func = self.__rand_sin(full_episode=True)
        prices.append(p)
        funcs.append(func)
        return np.array(prices).T, str(funcs)


def test_SinSampler():
    window_episode = 180
    window_state = 40
    noise_amplitude_ratio = 0.5
    period_range = (10, 40)
    amplitude_range = (5, 80)
    game = 'concat_half_base'
    instruments = ['fake']

    sampler = SinSampler(game,
                         window_episode, noise_amplitude_ratio, period_range, amplitude_range)
    n_episodes = 100
    """
    for i in range(100):
        plt.plot(sampler.sample(instruments)[0])
        plt.show()
        """
    fld = os.path.join('data', 'SinSamplerDB', game + '_B')
    sampler.build_db(n_episodes, fld)


if __name__ == '__main__':
    # scan_match()
    test_SinSampler()
    # p = [1,2,3,2,1,2,3]
    # print find_ideal(p)
    test_PairSampler()
