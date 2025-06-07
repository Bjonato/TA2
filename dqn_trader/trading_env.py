import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """Trading environment for OHLCV data with buy/sell/hold actions.

    The agent can open multiple positions up to ``position_limit`` and pays a
    small ``commission`` on each buy and sell. When the episode ends, any
    remaining positions are closed automatically. A penalty is applied if the
    total number of completed trades falls outside the range specified by
    ``min_trades`` and ``max_trades``.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, commission: float = 0.0005,
                 min_trades: int = 1, max_trades: int | None = None,
                 position_limit: int = 5, trade_penalty: float = 1.0):
        super().__init__()
        assert set(['Open', 'High', 'Low', 'Close', 'Volume']).issubset(df.columns), 'Missing OHLCV columns'
        self.df = df.reset_index(drop=True)
        self.commission = commission
        self.min_trades = min_trades
        self.position_limit = position_limit
        self.max_trades = max_trades
        self.trade_penalty = trade_penalty
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        self.positions = []
        self.total_trades = 0
        self.wins = 0
        self.total_reward = 0.0
        self.risk_rewards = []
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.loc[self.index]
        obs = row[['Open', 'High', 'Low', 'Close', 'Volume']].astype(np.float32).to_numpy()
        return obs

    def step(self, action):
        done = False
        price = self.df.loc[self.index, 'Close']
        reward = 0.0

        if action == 1 and len(self.positions) < self.position_limit:  # buy
            self.positions.append(price * (1 + self.commission))
        elif action == 2 and self.positions:
            buy_price = self.positions.pop(0)
            self.total_trades += 1
            pnl = price * (1 - self.commission) - buy_price
            reward += pnl
            self.total_reward += pnl
            if pnl > 0:
                self.wins += 1
            if buy_price != 0:
                self.risk_rewards.append(pnl / abs(buy_price - price))

        self.index += 1
        if self.index >= len(self.df) - 1:
            done = True
            # close remaining positions
            while self.positions:
                buy_price = self.positions.pop(0)
                pnl = price * (1 - self.commission) - buy_price
                self.total_trades += 1
                reward += pnl
                self.total_reward += pnl
                if pnl > 0:
                    self.wins += 1
                if buy_price != 0:
                    self.risk_rewards.append(pnl / abs(buy_price - price))
            if self.total_trades < self.min_trades:
                reward -= (self.min_trades - self.total_trades) * self.trade_penalty
            if self.max_trades is not None and self.total_trades > self.max_trades:
                reward -= (self.total_trades - self.max_trades) * self.trade_penalty
        obs = self._get_observation() if not done else np.zeros(5, dtype=np.float32)
        info = {
            'trades': self.total_trades,
            'wins': self.wins,
            'risk_reward': np.mean(self.risk_rewards) if self.risk_rewards else 0.0,
            'pnl': self.total_reward
        }
        return obs, reward, done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.index} Price: {self.df.loc[self.index, 'Close']} Positions: {len(self.positions)}")
