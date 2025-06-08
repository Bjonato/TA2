import argparse
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from .train import load_data
from .trading_env import TradingEnv


def main():
    parser = argparse.ArgumentParser(description="Test trained DQN trading agent")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--model", type=str, default="dqn_trading.zip", help="Path to trained model")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--position-limit", type=int, default=5)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--max-trades", type=int, default=None)
    parser.add_argument("--trade-penalty", type=float, default=1.0)
    parser.add_argument("--trade-bonus", type=float, default=0.0)
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
        )
    ])

    model = DQN.load(args.model)

    obs = env.reset()
    done = [False]
    info = {}
    actions = []
    prices = []
    dates = []

    while not done[0]:
        current_index = env.envs[0].index
        prices.append(env.envs[0].df.loc[current_index, "Close"])
        dates.append(env.envs[0].df.loc[current_index, "Dates"])
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0])
        obs, reward, done, infos = env.step(action)
        info = infos[0]

    win_rate = info['wins'] / info['trades'] if info['trades'] else 0
    print("Total Trades:", info['trades'])
    print("Win Rate:", win_rate)
    print("Average R/R:", info['risk_reward'])
    print("Total PnL:", info['pnl'])
    print("Biggest Win:", info['max_win'])
    print("Biggest Loss:", info['max_loss'])

    buy_x = [dates[i] for i, a in enumerate(actions) if a == 1]
    buy_y = [prices[i] for i, a in enumerate(actions) if a == 1]
    sell_x = [dates[i] for i, a in enumerate(actions) if a == 2]
    sell_y = [prices[i] for i, a in enumerate(actions) if a == 2]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, prices, label="Close")
    if buy_x:
        plt.scatter(buy_x, buy_y, marker='^', color='g', label='Buy')
    if sell_x:
        plt.scatter(sell_x, sell_y, marker='v', color='r', label='Sell')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Trading Decisions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
