import argparse
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .trading_env import TradingEnv

class TrainLogger(BaseCallback):
    """Callback printing evaluation statistics during training."""

    def __init__(self, eval_env, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.epoch = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.epoch += 1
            obs = self.eval_env.reset()
            done = False
            info = {}
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, infos = self.eval_env.step(action)
                info = infos[0]
            win_rate = info['wins'] / info['trades'] if info['trades'] else 0
            print(
                f"Epoch {self.epoch} - Step {self.n_calls}: "
                f"trades={info['trades']} win_rate={win_rate:.2f} "
                f"pnl={info['pnl']:.2f}"
            )
        return True

def load_data(path: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Load OHLCV data from CSV allowing optional preamble lines."""
    header_row = 0
    with open(path, "r") as f:
        for i, line in enumerate(f):
            lower = line.lower()
            if all(k in lower for k in ["open", "close", "high", "low", "volume"]):
                header_row = i
                break
    df = pd.read_csv(path, header=header_row)
    df = df.dropna()
    if 'Dates' in df.columns:
        df['Dates'] = pd.to_datetime(df['Dates'])
        if start_date:
            df = df[df['Dates'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Dates'] <= pd.to_datetime(end_date)]
        df = df.reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Train DQN trading agent")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--min_trades", type=int, default=1)
    parser.add_argument("--start-date", type=str, default=None, help="Filter data from this date (inclusive)")
    parser.add_argument("--end-date", type=str, default=None, help="Filter data up to this date (inclusive)")
    parser.add_argument("--eval-freq", type=int, default=1000, help="Steps between progress evaluations")
    args = parser.parse_args()

    df = load_data(args.data, start_date=args.start_date, end_date=args.end_date)
    env = DummyVecEnv([lambda: TradingEnv(df, min_trades=args.min_trades)])
    eval_env = DummyVecEnv([lambda: TradingEnv(df, min_trades=args.min_trades)])

    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="runs")
    model.learn(
        total_timesteps=args.timesteps,
        callback=TrainLogger(eval_env, eval_freq=args.eval_freq)
    )
    model.save("dqn_trading")

    # evaluate on the full dataset once more
    obs = eval_env.reset()
    done = False
    info = {}
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, infos = eval_env.step(action)
        info = infos[0]
    print("Total Trades:", info['trades'])
    win_rate = info['wins'] / info['trades'] if info['trades'] else 0
    print("Win Rate:", win_rate)
    print("Average R/R:", info['risk_reward'])
    print("Total PnL:", info['pnl'])

if __name__ == "__main__":
    main()

