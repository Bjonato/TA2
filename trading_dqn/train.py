import argparse
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
# allow execution from repository root
from .trading_env import TradingEnv

class TrainLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.epoch = 0

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print(f"Step: {self.n_calls}")
        return True


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def main():
    parser = argparse.ArgumentParser(description="Train DQN trading agent")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--min_trades", type=int, default=1)
    args = parser.parse_args()

    df = load_data(args.data)
    env = DummyVecEnv([lambda: TradingEnv(df, min_trades=args.min_trades)])

    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="runs")
    model.learn(total_timesteps=args.timesteps, callback=TrainLogger())
    model.save("dqn_trading")

    # evaluate
    obs = env.reset()
    done = False
    info = {}
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        info = infos[0]
    print("Total Trades:", info['trades'])
    win_rate = info['wins'] / info['trades'] if info['trades'] else 0
    print("Win Rate:", win_rate)
    print("Average R/R:", info['risk_reward'])
    print("Total PnL:", info['pnl'])

if __name__ == "__main__":
    main()
