import pandas as pd
import numpy as np
import numba

@numba.jit(nopython=True)
def _lee_ready_numba(price: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """
    Numba-optimized implementation of the Lee & Ready algorithm.

    Args:
        price: NumPy array of trade prices.
        bid: NumPy array of best bid prices.
        ask: NumPy array of best ask prices.

    Returns:
        NumPy array with inferred trade direction: 1 for buy, -1 for sell, 0 for unknown (edge cases).
    """
    n = len(price)
    direction = np.zeros(n, dtype=np.int8) # Use int8 for memory efficiency
    midpoint = (bid + ask) / 2.0
    price_change = np.diff(price) # Calculate differences once

    last_direction = 0 # Store the last non-zero direction for zero-tick rule

    for i in range(n):
        # --- Quote Rule ---
        if price[i] > midpoint[i]:
            direction[i] = 1
            last_direction = 1
        elif price[i] < midpoint[i]:
            direction[i] = -1
            last_direction = -1
        else:
            # --- Tick Test (Trade at Midpoint) ---
            if i == 0:
                # Cannot apply tick test to the first trade, leave as 0 or assign default?
                # Lee & Ready paper isn't explicit. Often ignored or assigned previous day's close tick.
                # We'll leave it as 0, to be potentially filled later if needed.
                direction[i] = 0
                # We don't update last_direction here, wait for the first classified trade
            else:
                change = price[i] - price[i-1] # Use pre-calculated diff? No, index mismatch. Calculate directly.
                # change = price_change[i-1] # Equivalent to price[i] - price[i-1]

                if change > 0: # Uptick
                    direction[i] = 1
                    last_direction = 1
                elif change < 0: # Downtick
                    direction[i] = -1
                    last_direction = -1
                else: # Zero-tick
                    # Use the direction of the *last trade whose direction was determined*
                    direction[i] = last_direction
                    # last_direction remains unchanged

    # Optional: Handle any leading zeros if the first few trades couldn't be classified
    # E.g., if the first trade is at midpoint, direction[0] is 0.
    # A backfill strategy might be suitable here if needed, but requires another pass.
    # For simplicity and performance, we return the array as is.
    # The user can apply bfill outside if necessary.

    return direction


def infer_trade_direction(trades_df: pd.DataFrame, price_col: str = 'price', bid_col: str = 'bid', ask_col: str = 'ask') -> pd.Series:
    """
    Infers trade direction using the Lee & Ready (1991) algorithm (Optimized with Numba).

    Args:
        trades_df: DataFrame containing tick data, sorted by time.
                   Must include trade price, best bid, and best ask columns.
        price_col: Name of the column containing trade prices.
        bid_col: Name of the column containing best bid prices.
        ask_col: Name of the column containing best ask prices.

    Returns:
        A Pandas Series with inferred trade direction: 1 for buy, -1 for sell.
        Potentially contains 0 for initial trades if they occur at the midpoint before
        the first classifiable trade.
    """
    # Ensure required columns exist
    if not all(col in trades_df.columns for col in [price_col, bid_col, ask_col]):
        raise ValueError(f"DataFrame must contain columns: {price_col}, {bid_col}, {ask_col}")

    # Extract data as NumPy arrays for Numba
    # Using .values or .to_numpy() - .to_numpy() is preferred
    price = trades_df[price_col].to_numpy(dtype=np.float64)
    bid = trades_df[bid_col].to_numpy(dtype=np.float64)
    ask = trades_df[ask_col].to_numpy(dtype=np.float64)

    # Call the Numba-optimized function
    direction_arr = _lee_ready_numba(price, bid, ask)

    # Return as a Pandas Series with the original index
    return pd.Series(direction_arr, index=trades_df.index, name='lee_ready_direction')


# Example Usage (assuming you have a CSV or DataFrame named 'tick_data')
if __name__ == '__main__':
    # Create sample data
    data = {
        'timestamp': pd.to_datetime([
            '2023-01-01 09:30:01', '2023-01-01 09:30:02', '2023-01-01 09:30:03',
            '2023-01-01 09:30:04', '2023-01-01 09:30:05', '2023-01-01 09:30:06',
            '2023-01-01 09:30:07', '2023-01-01 09:30:08', '2023-01-01 09:30:09'
        ]),
        'price': [100.10, 100.05, 100.05, 100.06, 100.06, 100.05, 100.05, 100.05, 100.07],
        'bid':   [100.00, 100.00, 100.00, 100.05, 100.05, 100.04, 100.04, 100.04, 100.06],
        'ask':   [100.10, 100.10, 100.10, 100.07, 100.07, 100.06, 100.06, 100.06, 100.08]
    }
    tick_data = pd.DataFrame(data)
    tick_data = tick_data.set_index('timestamp') # Optional: Set timestamp as index

    print("Sample Tick Data:")
    print(tick_data)
    print("-" * 30)

    # Infer directions
    inferred_directions = infer_trade_direction(tick_data)

    # Add directions to the DataFrame
    tick_data['lee_ready_direction'] = inferred_directions

    print("Tick Data with Inferred Directions (1=Buy, -1=Sell) [Numba Optimized]:")
    print(tick_data)

    # Explanation of results for the sample data:
    # 1: Price 100.10 > Midpoint 100.05 -> Buy (1)
    # 2: Price 100.05 == Midpoint 100.05 -> Tick test: Price 100.05 < Prev Price 100.10 (Downtick) -> Sell (-1)
    # 3: Price 100.05 == Midpoint 100.05 -> Tick test: Price 100.05 == Prev Price 100.05 (Zero-tick) -> Use previous direction (-1) -> Sell (-1)
    # 4: Price 100.06 == Midpoint 100.06 -> Tick test: Price 100.06 > Prev Price 100.05 (Uptick) -> Buy (1)
    # 5: Price 100.06 == Midpoint 100.06 -> Tick test: Price 100.06 == Prev Price 100.06 (Zero-tick) -> Use previous direction (1) -> Buy (1)
    # 6: Price 100.05 == Midpoint 100.05 -> Tick test: Price 100.05 < Prev Price 100.06 (Downtick) -> Sell (-1)
    # 7: Price 100.05 == Midpoint 100.05 -> Tick test: Price 100.05 == Prev Price 100.05 (Zero-tick) -> Use previous direction (-1) -> Sell (-1)
    # 8: Price 100.05 == Midpoint 100.05 -> Tick test: Price 100.05 == Prev Price 100.05 (Zero-tick) -> Use previous direction (-1) -> Sell (-1)
    # 9: Price 100.07 == Midpoint 100.07 -> Tick test: Price 100.07 > Prev Price 100.05 (Uptick) -> Buy (1)
