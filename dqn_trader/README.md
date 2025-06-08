# DQN Trader

This folder contains a simple Deep Q-Network (DQN) agent for day trading based on OHLCV data.

## Usage

1. Install requirements (already included in this repo):
   ```bash
   pip install stable-baselines3 gym pandas numpy ta tqdm matplotlib
   ```

2. Prepare your CSV data file. It may include a descriptive preamble before the column names (e.g. "BarTp,Trade,,," etc.). The loader automatically locates the header row and removes stray values such as `#NAME?` so the data can be parsed properly. Each record must provide `Dates`, `Open`, `Close`, `High`, `Low` and `Volume`.

3. Train the agent:
   ```bash
   python -m dqn_trader.train --data path/to/data.csv --timesteps 50000 --min_trades 30
   ```
   Additional options allow filtering by date range and setting how often training progress is printed:
   ```bash
    python -m dqn_trader.train \
        --data path/to/data.csv \
        --start-date 2024-10-01 \
        --end-date 2024-10-11 \
        --eval-freq 2000 \
        --commission 0.0005 \
       --position-limit 5 \
       --max-trades 300 \
       --trade-penalty 1.0 \
       --trade-bonus 0.01
       --enforce-min-trades \
       --max-passes 5
   ```
    The additional parameters allow you to control the commission paid on each
    trade, the maximum number of open positions, and the target range of trades.
    The environment penalizes the agent if it completes fewer than
    `min_trades` or exceeds `max_trades` in a single dataset pass. When
    `--enforce-min-trades` is supplied, the environment will automatically
    loop over the data up to `--max-passes` times until the minimum trade count
    is reached. A small `trade_bonus` can also reward each completed trade.
    Training
    progress prints epoch statistics every `eval-freq` steps. Key DQN
    hyperparameters such as learning rate and exploration settings can also be
    overridden via command line flags.

4. After training, the script evaluates the agent once and prints:
   - total trades
   - win rate
   - average risk/reward
   - total PnL

## Testing a saved model

Use the separate `test.py` script to evaluate the saved weights on a
different CSV file and visualise trading decisions:

```bash
python -m dqn_trader.test --data path/to/test.csv --model dqn_trading.zip
```
This command prints total trades, win rate, average risk/reward, total PnL,
and the biggest win and loss. A chart showing the price action with buy and
sell markers is displayed at the end.

The trained model is saved as `dqn_trading.zip` in the working directory.