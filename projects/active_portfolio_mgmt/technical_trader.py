import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
import logging
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HMMPairTrader:
    """
    Implements a pair trading strategy using a Hidden Markov Model (HMM)
    to model the spread between two assets.
    """
    def __init__(self, asset1: str, asset2: str, start_date: str, end_date: str, n_states: int = 3, random_state: int = 42):
        """
        Initializes the HMMPairTrader.

        Args:
            asset1 (str): Ticker symbol for the first asset.
            asset2 (str): Ticker symbol for the second asset.
            start_date (str): Start date for historical data (YYYY-MM-DD).
            end_date (str): End date for historical data (YYYY-MM-DD).
            n_states (int): The number of hidden states in the HMM.
            random_state (int): Random seed for reproducibility.
        """
        self.asset1 = asset1
        self.asset2 = asset2
        self.start_date = start_date
        self.end_date = end_date
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.data = None
        self.spread = None
        self.hidden_states = None

        logging.info(f"Initialized HMMPairTrader for {asset1}-{asset2} from {start_date} to {end_date} with {n_states} states.")

    def _fetch_data(self) -> None:
        """Fetches historical OHLCV data for the two assets."""
        logging.info(f"Fetching data for {self.asset1} and {self.asset2}...")
        try:
            # Fetch full OHLCV data
            data_asset1 = yf.download(self.asset1, start=self.start_date, end=self.end_date)
            data_asset2 = yf.download(self.asset2, start=self.start_date, end=self.end_date)

            if data_asset1.empty or data_asset2.empty:
                 raise ValueError("No data fetched for one or both assets.")

            # Use Adj Close for spread calculation, but keep OHLCV for backtesting
            self.adj_close_data = pd.DataFrame({
                self.asset1: data_asset1['Adj Close'],
                self.asset2: data_asset2['Adj Close']
            }).dropna()

            # Prepare data for backtesting.py (needs Open, High, Low, Close, Volume)
            # We'll use asset1's data as the primary instrument for the backtest framework
            self.ohlcv_data = data_asset1[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            # Align indices after potential drops
            common_index = self.adj_close_data.index.intersection(self.ohlcv_data.index)
            self.adj_close_data = self.adj_close_data.loc[common_index]
            self.ohlcv_data = self.ohlcv_data.loc[common_index]


            if self.adj_close_data.empty or self.ohlcv_data.empty:
                raise ValueError("Data alignment resulted in empty DataFrame.")

            logging.info("OHLCV and Adjusted Close data fetched successfully.")
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            self.data = None # Ensure data is None if fetch fails
            self.adj_close_data = None
            self.ohlcv_data = None
            raise

    def _calculate_spread(self) -> None:
        """Calculates the spread between the two assets using Adjusted Close prices."""
        if self.adj_close_data is None or self.adj_close_data.empty:
            logging.error("Adjusted Close data not available. Cannot calculate spread.")
            return
        # Using log price ratio as spread is common
        self.spread = np.log(self.adj_close_data[self.asset1] / self.adj_close_data[self.asset2])
        self.spread = self.spread.values.reshape(-1, 1) # Reshape for HMM
        logging.info("Spread calculated.")

    def _train_hmm(self) -> None:
        """Trains the Gaussian HMM on the calculated spread."""
        if self.spread is None:
            logging.error("Spread not calculated. Cannot train HMM.")
            return

        logging.info(f"Training Gaussian HMM with {self.n_states} states...")
        self.model = hmm.GaussianHMM(n_components=self.n_states,
                                     covariance_type="diag", # Diagonal covariance matrix
                                     n_iter=1000,           # Number of iterations
                                     random_state=self.random_state)
        try:
            self.model.fit(self.spread)
            if not self.model.monitor_.converged:
                logging.warning("HMM training did not converge.")
            else:
                logging.info("HMM training converged.")
        except Exception as e:
            logging.error(f"Error during HMM training: {e}")
            raise

    def _predict_states(self) -> None:
        """Predicts the sequence of hidden states for the observed spread."""
        if self.model is None:
            logging.error("HMM model not trained. Cannot predict states.")
            return
        if self.spread is None:
             logging.error("Spread not available. Cannot predict states.")
             return

        try:
            self.hidden_states = self.model.predict(self.spread)
            logging.info("Hidden states predicted.")
        except Exception as e:
            logging.error(f"Error predicting hidden states: {e}")
            raise

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates trading signals based on the predicted hidden states.
        """
        if self.hidden_states is None:
            logging.error("Hidden states not predicted. Cannot generate signals.")
            return pd.DataFrame()
        # Use adj_close_data for prices in signals df now
        if self.adj_close_data is None or self.adj_close_data.empty:
             logging.error("Adjusted close price data not available. Cannot generate signals DataFrame.")
             return pd.DataFrame()

        signals = pd.DataFrame(index=self.adj_close_data.index)
        signals['Price1'] = self.adj_close_data[self.asset1]
        signals['Price2'] = self.adj_close_data[self.asset2]
        # Ensure spread aligns with the current index before assigning
        spread_series = pd.Series(self.spread.flatten(), index=self.adj_close_data.index[:len(self.spread)])
        signals['Spread'] = spread_series
        signals['Hidden_State'] = self.hidden_states # Assuming hidden_states aligns with spread

        signals = signals.dropna() # Drop rows where spread/state might be missing if lengths mismatch

        # --- Placeholder Trading Logic ---
        # Assign meaning to states based on their mean spread value in the HMM model
        state_means = self.model.means_.flatten()
        state_order = np.argsort(state_means) # Order states by mean spread (low to high)
        mean_reverting_state = state_order[self.n_states // 2] # Assume middle state is mean-reverting
        lower_threshold_state = state_order[0] # Lowest mean spread state
        upper_threshold_state = state_order[-1] # Highest mean spread state

        logging.info(f"Interpreting states (ordered by mean spread): {state_order}")
        logging.info(f"State Means: {state_means[state_order]}")
        logging.info(f"Assumed Mean-Reverting State: {mean_reverting_state}")
        logging.info(f"Assumed Lower-Threshold State: {lower_threshold_state}")
        logging.info(f"Assumed Upper-Threshold State: {upper_threshold_state}")

        signals['Position'] = 0 # Initialize Position column

        # --- Refined Trading Logic ---
        # Enter Short when in the upper state (expecting reversion down).
        # Enter Long when in the lower state (expecting reversion up).
        # Exit when reaching the middle (mean-reverting) state.
        # Exit also if state transitions back to the initial extreme state (failed reversion).

        for i in range(1, len(signals)):
            current_state = signals['Hidden_State'].iloc[i]
            prev_position = signals['Position'].iloc[i-1]

            # By default, maintain the previous position
            signals['Position'].iloc[i] = prev_position

            # --- Entry Logic ---
            # Enter Short Spread (Sell Asset1, Buy Asset2) if currently flat
            if current_state == upper_threshold_state and prev_position == 0:
                 signals['Position'].iloc[i] = -1
                 logging.debug(f"{signals.index[i]}: Enter Short Spread (State: {current_state}, Prev Pos: {prev_position})")
            # Enter Long Spread (Buy Asset1, Sell Asset2) if currently flat
            elif current_state == lower_threshold_state and prev_position == 0:
                 signals['Position'].iloc[i] = 1
                 logging.debug(f"{signals.index[i]}: Enter Long Spread (State: {current_state}, Prev Pos: {prev_position})")

            # --- Exit Logic ---
            # Exit any position if reaching the mean-reverting state
            elif current_state == mean_reverting_state and prev_position != 0:
                 signals['Position'].iloc[i] = 0
                 logging.debug(f"{signals.index[i]}: Exit to Flat (State: {current_state}, Prev Pos: {prev_position})")

            # Exit Short if it reverts back to upper state (failed reversion)
            elif current_state == upper_threshold_state and prev_position == -1:
                 # This condition might seem counter-intuitive, but prevents holding
                 # if the spread widens further instead of reverting.
                 # We only entered short *expecting* it to leave this state downwards.
                 # If it stays or returns, our mean-reversion assumption is failing here.
                 # signals['Position'].iloc[i] = 0 # Optional: Could exit here
                 # Keep holding for now, exit relies on hitting mean state. Alternate view below:
                 pass # Keep holding Short, main exit is hitting mean state.

            # Exit Long if it reverts back to lower state (failed reversion)
            elif current_state == lower_threshold_state and prev_position == 1:
                 # Similar logic to the short side.
                 # signals['Position'].iloc[i] = 0 # Optional: Could exit here
                 pass # Keep holding Long, main exit is hitting mean state.


        # Simple signal based on change in position: 1 = buy, -1 = sell, 0 = hold
        # Note: This 'Signal' column is different from the position itself.
        # It represents the *action* taken at the start of the day.
        signals['Signal'] = signals['Position'].diff().fillna(0)


        logging.info("Trading signals and positions generated.")
        return signals

    def run_strategy(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Executes the full strategy: fetch, calculate, train, predict, signal.
        Returns the OHLCV data and the signals DataFrame.
        """
        try:
            self._fetch_data()
        except Exception as e:
            logging.error(f"Strategy execution failed during data fetch: {e}")
            return None, None

        if self.ohlcv_data is None or self.adj_close_data is None:
             return None, None # Return empty if data fetching failed

        self._calculate_spread()
        if self.spread is None:
            return self.ohlcv_data, pd.DataFrame() # Return OHLCV but empty signals

        self._train_hmm()
        if self.model is None:
            return self.ohlcv_data, pd.DataFrame()

        self._predict_states()
        if self.hidden_states is None:
            return self.ohlcv_data, pd.DataFrame()

        signals_df = self.generate_signals()

        # Align signals_df index with ohlcv_data index before returning
        common_index = self.ohlcv_data.index.intersection(signals_df.index)
        self.ohlcv_data = self.ohlcv_data.loc[common_index]
        signals_df = signals_df.loc[common_index]

        return self.ohlcv_data, signals_df

# --- Backtesting Strategy ---
class HMMPairTradingStrategy(Strategy):
    """
    Backtesting.py Strategy class for the HMM Pair Trader signals.
    """
    # Pass signals DataFrame during initialization
    hmm_signals: pd.DataFrame = None

    def init(self):
        # Ensure signals are available and aligned with backtest data
        if self.hmm_signals is None:
            raise ValueError("HMMPairTrader signals must be provided to the strategy.")

        # Align signals with the backtest data index
        self.hmm_signals = self.hmm_signals.reindex(self.data.index).ffill() # Forward fill signals for missing days

        # Create a signal series accessible by the strategy
        self.signal = self.I(lambda x: x, self.hmm_signals['Signal'], name='HMM_Signal')
        self.position_signal = self.I(lambda x: x, self.hmm_signals['Position'], name='HMM_Position') # Use position for logic

    def next(self):
        # Get the target position for the current step from pre-calculated signals
        target_position = self.position_signal[-1] # Position signal tells us desired state (1: Long, -1: Short, 0: Flat)
        current_position_size = self.position.size # backtesting.py position size (shares)

        # Simple logic: Go long spread if signal is 1, short if -1, flat if 0
        # Note: backtesting.py handles sizing. We just tell it to buy or sell.
        # We are abstracting the pair trade into a single "instrument" (asset1) action.
        # Long spread = buy asset1 (implicitly short asset2)
        # Short spread = sell asset1 (implicitly long asset2)

        if target_position == 1 and not self.position: # If target is Long and currently flat
            self.buy() # Go Long Spread
        elif target_position == -1 and not self.position: # If target is Short and currently flat
             self.sell() # Go Short Spread
        elif target_position == 0 and self.position: # If target is Flat and currently have a position
             self.position.close() # Close existing position


# Example Usage (with Backtesting)
if __name__ == "__main__":
    # --- Parameters ---
    asset_1 = 'GLD'  # Example: Gold ETF
    asset_2 = 'GDX'  # Example: Gold Miners ETF
    start = '2015-01-01'
    end = '2023-12-31'
    num_states = 3   # Low Spread, Mean Reverting, High Spread

    # --- Initialize and Run HMM Strategy Calculation ---
    trader = HMMPairTrader(asset1=asset_1, asset2=asset_2, start_date=start, end_date=end, n_states=num_states)
    ohlcv_data, signals_df = trader.run_strategy()

    if ohlcv_data is not None and signals_df is not None and not signals_df.empty:
        logging.info("HMM Signal Generation Complete. Starting Backtest.")
        print("Signals DataFrame Head:")
        print(signals_df.head())

        # --- Prepare Data for Backtesting.py ---
        # Ensure signals_df is aligned with ohlcv_data (already done in run_strategy)
        # The backtest will run on ohlcv_data (using asset1's OHLCV)
        backtest_data = ohlcv_data.copy()

        # --- Run Backtest ---
        # Pass the generated signals to the strategy class
        HMMPairTradingStrategy.hmm_signals = signals_df

        bt = Backtest(
            backtest_data,           # Data (Asset 1 OHLCV)
            HMMPairTradingStrategy,  # Strategy class
            cash=100_000,            # Initial cash
            commission=.002,         # Example commission
            exclusive_orders=True    # Ensure one order at a time
        )

        stats = bt.run()
        logging.info("Backtest Finished.")
        print("\nBacktesting Statistics:")
        print(stats)

        # --- Optional: Plot Backtest Results ---
        # Backtesting.py uses Bokeh for plotting
        try:
            bt.plot()
            logging.info("Backtest plot generated (check browser window).")
        except Exception as e:
            logging.error(f"Could not generate backtest plot: {e}")

        # --- Optional: Plot Original Signals (requires matplotlib) ---
        # This uses the previous plotting logic for HMM states/spread
        if trader.model is not None:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.lines import Line2D

                fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
                fig.suptitle(f'HMM Pair Strategy Analysis: {asset_1} vs {asset_2}', fontsize=16)

                # Plot prices (using Adj Close from signals_df)
                axes[0].plot(signals_df.index, signals_df['Price1'], label=trader.asset1)
                axes[0].plot(signals_df.index, signals_df['Price2'], label=trader.asset2)
                axes[0].set_title(f'{trader.asset1} vs {trader.asset2} Adjusted Close Prices')
                axes[0].legend()
                axes[0].grid(True)
                axes[0].set_ylabel("Price")

                # Plot Spread and Hidden States
                state_means = trader.model.means_.flatten()
                state_order = np.argsort(state_means)
                colors = plt.cm.viridis(np.linspace(0, 1, trader.n_states))
                state_color_map = {state: colors[list(state_order).index(state)] for state in range(trader.n_states)}

                for i in range(len(signals_df) - 1):
                     state = int(signals_df['Hidden_State'].iloc[i]) # Ensure state is int
                     axes[1].plot(signals_df.index[i:i+2], signals_df['Spread'].iloc[i:i+2], color=state_color_map[state])

                legend_elements = [Line2D([0], [0], color=state_color_map[state], lw=4, label=f'State {state} (Mean: {trader.model.means_[state][0]:.3f})')
                                   for state in state_order]
                axes[1].set_title('Log Spread and HMM Hidden States')
                axes[1].legend(handles=legend_elements, loc='upper right')
                axes[1].grid(True)
                axes[1].set_ylabel("Log Spread")

                # Plot Positions (from signals_df)
                axes[2].plot(signals_df.index, signals_df['Position'], drawstyle='steps-post')
                axes[2].set_title('Target Trading Position (1: Long Spread, -1: Short Spread, 0: Flat)')
                axes[2].set_yticks([-1, 0, 1])
                axes[2].grid(True)
                axes[2].set_xlabel("Date")
                axes[2].set_ylabel("Position")

                plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for suptitle
                plt.show()

            except ImportError:
                logging.warning("Matplotlib not installed. Skipping HMM signal plotting.")
            except Exception as e:
                 logging.error(f"Error during Matplotlib HMM plotting: {e}")
        else:
             logging.warning("HMM model not available, skipping signal plots.")

    else:
        logging.warning("Strategy did not produce valid OHLCV data or signals. Skipping backtest.")
