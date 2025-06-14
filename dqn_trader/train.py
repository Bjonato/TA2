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
            done = [False]
            info = {}
            while not done[0]:
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

    # Clean up potential artefacts such as '#NAME?' rows
    if 'Dates' in df.columns:
        df = df[df['Dates'].astype(str).str.lower() != '#name?']
        df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
    numeric_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Dates'] + numeric_cols)

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
    parser.add_argument("--min_trades", type=int, default=30)
    parser.add_argument("--max-trades", type=int, default=300)
    parser.add_argument("--trade-penalty", type=float, default=1.0)
    parser.add_argument("--trade-bonus", type=float, default=0.0)
    parser.add_argument("--enforce-min-trades", action="store_true",
                        help="Extend episode until min_trades is met")
    parser.add_argument("--max-passes", type=int, default=5,
                        help="Maximum dataset passes when enforcing trades")
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--position-limit", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--exploration-final-eps", type=float, default=0.02)
    parser.add_argument("--start-date", type=str, default=None, help="Filter data from this date (inclusive)")
    parser.add_argument("--end-date", type=str, default=None, help="Filter data up to this date (inclusive)")
    parser.add_argument("--eval-freq", type=int, default=1000, help="Steps between progress evaluations")
    args = parser.parse_args()

    df = load_data(args.data, start_date=args.start_date, end_date=args.end_date)
    env = DummyVecEnv([
        lambda: TradingEnv(
            df,
            commission=args.commission,
            min_trades=args.min_trades,
            max_trades=args.max_trades,
            position_limit=args.position_limit,
            trade_penalty=args.trade_penalty,
            trade_bonus=args.trade_bonus,
            force_min_trades=args.enforce_min_trades,
            max_passes=args.max_passes,
        )
    ])
    eval_env = DummyVecEnv([
        lambda: TradingEnv(
            df,
            commission=args.commission,
            min_trades=args.min_trades,
            max_trades=args.max_trades,
            position_limit=args.position_limit,
            trade_penalty=args.trade_penalty,
            trade_bonus=args.trade_bonus,
            force_min_trades=args.enforce_min_trades,
            max_passes=args.max_passes,
        )
    ])

    model = DQN(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log="runs",
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=TrainLogger(eval_env, eval_freq=args.eval_freq)
    )
    model.save("dqn_trading")

    # evaluate on the full dataset once more
    obs = eval_env.reset()
    done = [False]
    info = {}
    while not done[0]:
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

