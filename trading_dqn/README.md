# Trading DQN

This folder contains a simple DQN day trading agent trained on OHLCV data.

## Usage

1. Install requirements (already included in this repo):
   ```bash
   pip install stable-baselines3 gym pandas numpy ta tqdm
   ```

2. Prepare your CSV data file.  It may include a few descriptive lines before the
   actual column names (e.g. "BarTp,Trade,,," etc.).  The loader scans for the
   first line containing the columns `Dates`, `Open`, `Close`, `High`, `Low` and
   `Volume` and reads the data from there.

3. Train the agent (run as a module):
   ```bash
   python -m trading_dqn.train --data path/to/data.csv --timesteps 50000 --min_trades 5
   ```
   Training progress prints the current step every 1000 iterations.

4. After training, the script evaluates the agent once and prints:
   - total trades
   - win rate
   - average risk/reward
   - total PnL

The model is saved as `dqn_trading.zip` in the working directory.
