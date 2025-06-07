# DQN Trader

This folder contains a simple Deep Q-Network (DQN) agent for day trading based on OHLCV data.

## Usage

1. Install requirements (already included in this repo):
   ```bash
   pip install stable-baselines3 gym pandas numpy ta tqdm
   ```

2. Prepare your CSV data file. It may include a descriptive preamble before the column names (e.g. "BarTp,Trade,,," etc.). The loader will automatically locate the line containing `Dates`, `Open`, `Close`, `High`, `Low` and `Volume`.

3. Train the agent:
   ```bash
   python -m dqn_trader.train --data path/to/data.csv --timesteps 50000 --min_trades 5
   ```
   Additional options allow filtering by date range and setting how often training progress is printed:
   ```bash
   python -m dqn_trader.train \
       --data path/to/data.csv \
       --start-date 2024-10-01 \
       --end-date 2024-10-11 \
       --eval-freq 2000
   ```
   Training progress prints epoch statistics every `eval-freq` steps.

4. After training, the script evaluates the agent once and prints:
   - total trades
   - win rate
   - average risk/reward
   - total PnL

The trained model is saved as `dqn_trading.zip` in the working directory.

