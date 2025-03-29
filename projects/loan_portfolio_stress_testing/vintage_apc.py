# Vintage Analysis Model for Credit Risk Forecasting
# ===============================================
#
# This code implements the Age-Period-Cohort (APC) model for credit risk stress testing
# as described in Bellotti and Crook (2015). The model is designed specifically for
# analyzing credit portfolio performance over time.
#
# Key Credit Risk Concepts:
# -------------------------
# 1. Age Effect (F) - How delinquency/default behavior changes as loans mature
#    (e.g., typically high in months 6-18, then stabilizing)
#
# 2. Vintage/Cohort Effect (G) - The inherent quality of loans originated in a specific period
#    (e.g., loans originated during loose underwriting periods perform worse)
#
# 3. Time/Period Effect (H) - Impact of current economic conditions on all loans
#    (e.g., increased unemployment raising defaults across all vintages)
#
# 4. Credit Score Effect - Added as a macroeconomic predictor to capture overall
#    creditworthiness trends in the borrower population
#
# The model enables:
# - Decomposing past performance into the three APC components
# - Attributing performance changes to specific factors (underwriting vs economy)
# - Stress testing portfolios under different economic scenarios
# - Long-term forecasting that accounts for loan aging patterns
#
# References:
# - Bellotti, A., & Crook, J. (2015). A vintage model of credit risk stress testing.
#   Journal of the Operational Research Society, 66(8), 1342-1352.
#   https://www.tandfonline.com/doi/full/10.1057/jors.2015.97

import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix, dmatrices # Ensure both are imported, especially dmatrices
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) # Patsy/Statsmodels future warnings
warnings.filterwarnings("ignore", category=UserWarning)   # Patsy user warnings about formulae

# --- Helper Functions ---
# (Keep helper functions as they were, rounding doesn't affect them directly)
def log_odds(rate, epsilon=1e-8):
    """Calculates log-odds, handling rates near 0 or 1."""
    rate = np.clip(rate, epsilon, 1 - epsilon)
    return np.log(rate / (1 - rate))

def inv_log_odds(logodds):
    """Calculates rate from log-odds."""
    return 1 / (1 + np.exp(-logodds))

def macro_log_ratio(series, lag, window):
    """Calculates log-ratio transformation for macro variable like HPI."""
    if len(series) < lag + window:
        return pd.Series(np.nan, index=series.index)
    numerator = series.shift(lag)
    denominator = series.shift(lag + window)
    ratio = numerator / (denominator + 1e-9)
    ratio[ratio <= 0] = np.nan
    logratio = np.log(ratio.replace([np.inf, -np.inf], np.nan))
    return logratio

def get_linear_trend(y, x):
    """Fits y = b0 + b1*x and returns the slope b1."""
    y = pd.Series(y).dropna()
    x = pd.Series(x).dropna()
    common_index = y.index.intersection(x.index)
    y = y[common_index]
    x = x[common_index]
    if len(x) < 2 or len(y) < 2:
        return 0.0
    X = sm.add_constant(x, has_constant='add')
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()
    if len(results.params) > 1 and X.shape[1] > 1:
        return results.params.iloc[1]
    else:
        return 0.0

# --- Data Simulation (Example) ---

def simulate_data(num_vintages=10, obs_per_vintage=40, start_year=2005, round_precision=5, 
                  macro_history_years=5):
    """
    Simulates vintage-level loan data and macroeconomic data for testing the model.
    
    This function creates realistic synthetic data that mimics:
    1. A portfolio with multiple loan vintages
    2. Aging patterns typical in consumer lending
    3. Vintage quality differences across origination periods
    4. Economic cycles and their impact on default rates
    5. Credit score trends that correlate with economic conditions
    
    Credit Risk Concepts in the Simulation:
    --------------------------------------
    - Default rates vary by loan age (high in early-mid life, then stabilizing)
    - Origination quality varies over time (vintage effect)
    - Economic conditions (unemployment, HPI) affect all loans
    - Credit scores trend with the economy but with a lag (credit deterioration follows unemployment)
    
    Parameters:
    -----------
    num_vintages : int
        Number of origination cohorts to simulate
    obs_per_vintage : int
        Maximum number of months to observe each vintage (loan age)
    start_year : int
        Year of the first vintage
    round_precision : int
        Decimal precision for rounding time values (prevents merge issues)
    macro_history_years : int
        Years of economic data to generate before first vintage
        
    Returns:
    --------
    tuple
        (portfolio_df, macro_short, macro_long)
        - portfolio_df: Vintage-level performance data
        - macro_short: Economic data matching portfolio period
        - macro_long: Extended economic history with pre-portfolio data
    """
    # Time grid - creating a consistent timeline for vintages and observations
    vintage_dates = pd.to_datetime([f'{start_year+i}-01-01' for i in range(num_vintages)])
    min_vintage_date = vintage_dates.min()
    max_obs_date = vintage_dates.max() + pd.DateOffset(months=obs_per_vintage - 1)
    df_dates = pd.DataFrame({'observation_date': pd.date_range(start=min_vintage_date, end=max_obs_date, freq='MS')})
    df_dates = df_dates.sort_values('observation_date').reset_index(drop=True)
    min_overall_date = df_dates['observation_date'].min()
    df_dates['time_numeric'] = round((df_dates['observation_date'] - min_overall_date).dt.days / 30.44, round_precision)

    # Portfolio Data - vintage-level loan performance
    portfolio_data = []
    for i, v_date in enumerate(vintage_dates):
        for m in range(obs_per_vintage):
            obs_date = v_date + pd.DateOffset(months=m)
            if obs_date > df_dates['observation_date'].max(): continue
            age = m
            time_num = round((obs_date - min_overall_date).days / 30.44, round_precision)
            vintage_num = round((v_date - min_overall_date).days / 30.44, round_precision)

            # Simulate default patterns with realistic components:
            
            # 1. Age effect: Typically rises early then stabilizes
            # (negative log shape plus small linear component)
            log_odds_age = -np.log(age + 1) * 0.5 + 0.02 * age - 1.0
            
            # 2. Vintage effect: Quality changes over time (cycles + trend)
            # (slight deterioration over time + cyclical pattern)
            log_odds_vintage = -0.05 * (vintage_num/12) + np.sin(vintage_num / 20) * 0.3
            
            # 3. Time effect: Economic impact on all loans
            # (cyclical pattern + slight upward trend)
            log_odds_time = np.cos(time_num / 15) * 0.5 + 0.01 * time_num
            
            # Combined effect with base rate and random noise
            true_log_odds = -2.5 + log_odds_age + log_odds_vintage + log_odds_time
            noise = np.random.normal(0, 0.2)  # Random variation between similar loans
            observed_log_odds = true_log_odds + noise

            portfolio_data.append({
                'vintage_date': v_date, 
                'observation_date': obs_date, 
                'age_months': age,
                'vintage_numeric': vintage_num, 
                'time_numeric': time_num,
                'log_odds_default': observed_log_odds
            })
    portfolio_df = pd.DataFrame(portfolio_data)

    # Get unique time points from portfolio data for consistent macro alignment
    portfolio_time_points = np.sort(portfolio_df['time_numeric'].unique())
    
    # Macro Data (Short history matching portfolio)
    # Use exactly the same time points as in portfolio for perfect alignment
    macro_short = pd.DataFrame({'time_numeric': portfolio_time_points})
    
    # Generate macro variables with consistent seed for reproducibility
    np.random.seed(42)
    
    # Unemployment rate: cyclical with upward trend + noise
    # Values typically range from 3-10% over economic cycles
    macro_short['unemployment'] = 5.0 + 3.0 * np.sin(macro_short['time_numeric'] / 15 + 1) + \
                               0.03 * macro_short['time_numeric'] + \
                               np.random.normal(0, 0.5, size=len(macro_short))
    
    # Housing Price Index (HPI): long-term growth with cycles
    # Starting at 100 with roughly 5% annual growth + cycles
    macro_short['hpi'] = 100 * np.exp(0.005 * macro_short['time_numeric'] + \
                               0.1 * np.cos(macro_short['time_numeric'] / 20) + \
                               np.random.normal(0, 0.02, size=len(macro_short)))

    # Macro Data (Long history - includes pre-portfolio period)
    # Create extended history for proper lag calculations
    start_time_long = -macro_history_years * 12  # Pre-history time points
    end_time_long = max(portfolio_time_points) + 1
    
    # Create continuous, evenly spaced grid of time points
    time_numeric_long = np.arange(start_time_long, end_time_long, 1.0)
    time_numeric_long = np.round(time_numeric_long, round_precision)
    
    # Add exact portfolio time points to ensure perfect alignment
    time_numeric_long = np.sort(np.unique(np.append(time_numeric_long, portfolio_time_points)))
    
    macro_long = pd.DataFrame({'time_numeric': time_numeric_long})
    
    # Use the same random seed for reproducibility
    np.random.seed(42)
    
    # Unemployment - same pattern but extended back in time
    macro_long['unemployment'] = 5.0 + 3.0 * np.sin(macro_long['time_numeric'] / 15 + 1) + \
                              0.03 * macro_long['time_numeric'] + \
                              np.random.normal(0, 0.5, size=len(macro_long))
    
    # Add a long-term trend to unemployment that's only visible in the longer history
    macro_long['unemployment'] += 0.005 * (macro_long['time_numeric'] - start_time_long)
    
    # HPI - same pattern but extended back in time
    macro_long['hpi'] = 100 * np.exp(0.005 * macro_long['time_numeric'] + \
                              0.1 * np.cos(macro_long['time_numeric'] / 20) + \
                              np.random.normal(0, 0.02, size=len(macro_long)))

    # Credit Score as a new macro covariate
    # -------------------------------------------------------
    # Average credit scores typically range from 660-740 in a population
    # They tend to lag economic changes (deteriorate after unemployment rises)
    np.random.seed(43)  # Different seed for some independence
    
    # Base credit score starting around 700 (average FICO)
    # With cyclical pattern and slight long-term trend
    macro_long['avg_credit_score'] = 700 + \
                                  -8.0 * np.sin(macro_long['time_numeric'] / 18 + 1.5) + \
                                  5.0 * np.cos(macro_long['time_numeric'] / 24) + \
                                  0.05 * macro_long['time_numeric'] + \
                                  np.random.normal(0, 2.0, size=len(macro_long))
                                  
    # Make credit scores slightly lagging the economy
    # (credit deterioration follows unemployment with a delay)
    macro_long['avg_credit_score'] = macro_long['avg_credit_score'].rolling(window=3, min_periods=1).mean()
    
    # Propagate credit scores to short macro data as well for consistency
    macro_short['avg_credit_score'] = np.interp(
        macro_short['time_numeric'],
        macro_long['time_numeric'],
        macro_long['avg_credit_score']
    )

    # Add observation dates for reference
    macro_short['observation_date'] = macro_short['time_numeric'].apply(lambda x: min_overall_date + pd.DateOffset(days=int(round(x*30.44))))
    macro_long['observation_date'] = macro_long['time_numeric'].apply(lambda x: min_overall_date + pd.DateOffset(days=int(round(x*30.44))))

    # Add true time effect for validation
    macro_long['true_log_odds_time'] = np.cos(macro_long['time_numeric'] / 15) * 0.5 + 0.01 * macro_long['time_numeric']
    
    # Print summary statistics
    print(f"Generated {len(portfolio_df)} portfolio observations across {num_vintages} vintages")
    print(f"Generated macro data with {len(macro_long)} time points ({-start_time_long/12:.1f} years history)")
    print(f"Portfolio time range: {portfolio_time_points.min()} to {portfolio_time_points.max()}")
    print(f"Macro time range: {macro_long['time_numeric'].min()} to {macro_long['time_numeric'].max()}")
    
    # Verify alignment
    if not set(portfolio_time_points).issubset(set(macro_long['time_numeric'])):
        print("WARNING: Portfolio time points not fully contained in macro data!")
    else:
        print("All portfolio time points are present in macro data (good alignment)")

    return portfolio_df, macro_short, macro_long


# --- Main Class ---
class APCStressTester:
    """
    Age-Period-Cohort (APC) model for credit risk stress testing and forecasting.
    
    This class implements the full APC model workflow:
    1. Decomposing historical default rates into age, vintage, and time effects
    2. Modeling the time effect using macroeconomic variables
    3. Calculating long-term trends and adjusting components
    4. Generating predictions under different economic scenarios
    
    The model helps credit risk managers:
    - Understand drivers of portfolio performance
    - Separate underwriting quality from economic effects
    - Create more accurate forecasts that account for portfolio aging
    - Perform stress testing under various economic scenarios
    """
    
    def __init__(self, portfolio_data, macro_data_long,
                 age_col='age_months', vintage_col='vintage_numeric', time_col='time_numeric',
                 target_col='log_odds_default',
                 ur_col='unemployment', hpi_col='hpi', credit_score_col='avg_credit_score',
                 spline_df_a=5, spline_df_v=5, spline_df_t=5,
                 ur_lag=3, ur_window=6,
                 hpi_lag=6, hpi_window=12,
                 credit_score_lag=2, credit_score_window=3,
                 round_precision=5):
        """
        Initialize the APC model with data and parameters.
        
        Parameters:
        -----------
        portfolio_data : pandas.DataFrame
            Loan-level or aggregated performance data with vintage and time dimensions
            Must contain columns for age, vintage, time, and target variable
        
        macro_data_long : pandas.DataFrame
            Macroeconomic data including history before portfolio start
            Must contain columns for time, unemployment, HPI, and credit scores
        
        age_col : str
            Name of column containing loan age (in months typically)
        
        vintage_col : str
            Name of column containing origination cohort identifier
            
        time_col : str
            Name of column containing observation time point
            
        target_col : str
            Name of column containing default rate or log-odds of default
            
        ur_col : str
            Name of column containing unemployment rate in macro data
            
        hpi_col : str
            Name of column containing housing price index in macro data
            
        credit_score_col : str
            Name of column containing average credit score in macro data
            
        spline_df_a, spline_df_v, spline_df_t : int
            Degrees of freedom for age, vintage, and time splines
            Higher values allow more flexibility/curvature in the components
            
        ur_lag : int
            Lag period (months) for unemployment rate effect
            Typical values: 1-6 months (credit cards react faster, mortgages slower)
            
        ur_window : int
            Window period (months) for smoothing unemployment rate
            Longer windows reduce noise but may miss turning points
            
        hpi_lag : int
            Lag period (months) for housing price index effect
            Typical values: 3-12 months (housing effects usually slower)
            
        hpi_window : int
            Window period (months) for HPI ratio calculation
            Typical values: 6-24 months (yearly or longer comparisons common)
            
        credit_score_lag : int
            Lag period (months) for credit score effect
            Typical values: 1-3 months (impacts usually quick)
            
        credit_score_window : int
            Window period (months) for smoothing credit scores
            Helps reduce noise in population credit score trends
            
        round_precision : int
            Decimal precision for internal time values
            Helps ensure consistent matching between datasets
        """
        # Store input data (make copies to avoid modifying originals)
        self.portfolio_data = portfolio_data.copy()
        self.macro_data_long = macro_data_long.copy()

        # Store column names
        self.age_col = age_col  # Loan age (months since origination)
        self.vintage_col = vintage_col  # Origination period
        self.time_col = time_col  # Observation time
        self.target_col = target_col  # Target variable (log-odds default)
        self.ur_col = ur_col  # Unemployment rate
        self.hpi_col = hpi_col  # Housing Price Index
        self.credit_score_col = credit_score_col  # Average credit score
        
        # Store model parameters
        self.spline_df_a = spline_df_a  # Controls flexibility of age curve
        self.spline_df_v = spline_df_v  # Controls flexibility of vintage curve
        self.spline_df_t = spline_df_t  # Controls flexibility of time curve

        # Macro variable transformation parameters
        self.ur_lag = ur_lag  # How many months to lag unemployment effect
        self.ur_window = ur_window  # Smoothing window for unemployment
        self.hpi_lag = hpi_lag  # How many months to lag HPI effect
        self.hpi_window = hpi_window  # Comparison period for HPI
        self.credit_score_lag = credit_score_lag  # How many months to lag credit score effect
        self.credit_score_window = credit_score_window  # Smoothing window for credit scores
        self.round_precision = round_precision  # Precision for avoiding floating point issues

        # Attributes to store results (populated during fitting)
        self.initial_decomp = {}  # Initial component decomposition
        self.macro_fit_results = None  # Regression results for macro model
        self.long_term_trend_eta = None  # Long-term trend parameter
        self.final_components = {}  # Final adjusted components
        self.model_intercept = None  # Overall intercept term
        self.macro_coeffs = None  # Coefficients for macro variables

        # Ensure time columns are rounded consistently before use
        # This helps avoid floating point precision issues during merges
        self.portfolio_data[self.time_col] = self.portfolio_data[self.time_col].round(self.round_precision)
        self.macro_data_long[self.time_col] = self.macro_data_long[self.time_col].round(self.round_precision)
        self.portfolio_data[self.vintage_col] = self.portfolio_data[self.vintage_col].round(self.round_precision)

        # Prepare macro data transformations
        self._prepare_macro_transforms()

    def _prepare_macro_transforms(self):
        """
        Pre-calculates transformed macroeconomic variables for model fitting.
        
        Each macro variable is transformed differently based on how it affects credit risk:
        
        1. Unemployment Rate (UR):
           - First converted to a rate (divide by 100)
           - Smoothed using a moving average (reduces noise)
           - Lagged to account for delayed impact on defaults
           - Transformed to log-odds scale (similar scale to target variable)
        
        2. Housing Price Index (HPI):
           - Calculated as log-ratio between current (lagged) and earlier period
           - Log transformation makes increases/decreases more symmetric
           - Negative values indicate housing price declines (higher risk)
           
        3. Credit Scores:
           - Smoothed using a moving average (reduces population noise)
           - Lagged to account for delayed impact
           - Standardized to z-score (mean 0, std 1) for comparability
           - Higher values indicate better creditworthiness (lower risk)
        
        The transformations help ensure:
        - Variables are on appropriate scales for regression
        - Timing of effects matches reality (through lags)
        - Relationships with default are more linear
        """
        # UR: Log-odds of moving average (unemployment as probability)
        # First clip to ensure valid probability range before log-odds transform
        ur_rate = np.clip(self.macro_data_long[self.ur_col] / 100.0, 1e-6, 1.0 - 1e-6)
        
        # Apply smoothing window to reduce noise (e.g., 3-month average)
        ur_smooth = ur_rate.rolling(window=self.ur_window, min_periods=1).mean()
        
        # Apply lag to account for delayed impact on defaults
        ur_lagged = ur_smooth.shift(self.ur_lag)
        
        # Transform to log-odds scale (matches default log-odds scale)
        self.macro_data_long['ur_transformed'] = log_odds(ur_lagged)

        # HPI: Log-ratio (current vs. previous period)
        # Log-ratio captures relative change (% change on log scale)
        self.macro_data_long['hpi_transformed'] = macro_log_ratio(
            self.macro_data_long[self.hpi_col],
            lag=self.hpi_lag,
            window=self.hpi_window
        )
        
        # Credit Score: Z-score transformation (standardize and lag)
        if self.credit_score_col in self.macro_data_long.columns:
            # Apply smoothing window to reduce population noise
            credit_smooth = self.macro_data_long[self.credit_score_col].rolling(
                window=self.credit_score_window, min_periods=1).mean()
            
            # Apply lag to account for delayed impact on defaults
            credit_lagged = credit_smooth.shift(self.credit_score_lag)
            
            # Standardize to z-score (mean 0, std 1) for better comparability with other variables
            credit_mean = credit_lagged.mean()
            credit_std = credit_lagged.std()
            
            if credit_std > 0:
                self.macro_data_long['credit_score_transformed'] = (credit_lagged - credit_mean) / credit_std
            else:
                # Fallback if std is 0 (unlikely but handle gracefully)
                self.macro_data_long['credit_score_transformed'] = 0
        else:
            # If credit score not available, create a dummy variable of zeros
            print(f"Warning: Credit score column '{self.credit_score_col}' not found in macro data.")
            self.macro_data_long['credit_score_transformed'] = 0

        # Add time numeric as a potential predictor in macro model
        # Captures any remaining time trend not explained by macro variables
        self.macro_data_long['time_trend_col'] = self.macro_data_long[self.time_col]


    def _initial_decomposition(self):
        """Step 1: Fit y ~ cr(a) + cr(v) + cr(t) and extract components."""

        min_df = 2
        n_unique_a = self.portfolio_data[self.age_col].nunique()
        n_unique_v = self.portfolio_data[self.vintage_col].nunique()
        n_unique_t = self.portfolio_data[self.time_col].nunique()

        df_a = max(min_df, min(self.spline_df_a, n_unique_a - 2 if n_unique_a > min_df else min_df))
        df_v = max(min_df, min(self.spline_df_v, n_unique_v - 2 if n_unique_v > min_df else min_df))
        df_t = max(min_df, min(self.spline_df_t, n_unique_t - 2 if n_unique_t > min_df else min_df))

        if df_a < self.spline_df_a or df_v < self.spline_df_v or df_t < self.spline_df_t:
             print(f"Warning: Effective spline df reduced. Requested ({self.spline_df_a},{self.spline_df_v},{self.spline_df_t}), Used ({df_a},{df_v},{df_t})")

        formula = f"{self.target_col} ~ cr({self.age_col}, df={df_a}) + cr({self.vintage_col}, df={df_v}) + cr({self.time_col}, df={df_t})"

        fit_data = self.portfolio_data[[self.target_col, self.age_col, self.vintage_col, self.time_col]].dropna()
        if fit_data.empty:
            raise ValueError("No valid data for initial decomposition after dropping NaNs.")

        try:
            y, X = dmatrices(formula, fit_data, return_type='dataframe')
        except Exception as e:
             print(f"Error during patsy formula processing: {e}"); raise

        ols_model = sm.OLS(y, X)
        ols_results = ols_model.fit()
        self.model_intercept = ols_results.params['Intercept']
        params = ols_results.params.drop('Intercept')

        data_for_pred = fit_data[[self.age_col, self.vintage_col, self.time_col]].copy()
        X_pred_full = dmatrix(X.design_info, data_for_pred, return_type='dataframe')

        di = X.design_info
        terms = di.term_names
        params_a_list, params_v_list, params_t_list = [], [], []
        for term in terms:
            if self.age_col in term: params_a_list.extend(list(X.columns[di.slice(term)]))
            elif self.vintage_col in term: params_v_list.extend(list(X.columns[di.slice(term)]))
            elif self.time_col in term: params_t_list.extend(list(X.columns[di.slice(term)]))

        params_a_list = sorted(list(set(params_a_list) - {'Intercept'}))
        params_v_list = sorted(list(set(params_v_list) - {'Intercept'}))
        params_t_list = sorted(list(set(params_t_list) - {'Intercept'}))

        if not params_a_list or not params_v_list or not params_t_list:
             raise ValueError("Failed to identify component parameters.")

        valid_params_a = [p for p in params_a_list if p in params.index]
        valid_params_v = [p for p in params_v_list if p in params.index]
        valid_params_t = [p for p in params_t_list if p in params.index]

        F_a_values = X_pred_full[valid_params_a] @ params[valid_params_a] if valid_params_a else 0.0
        G_v_values = X_pred_full[valid_params_v] @ params[valid_params_v] if valid_params_v else 0.0
        H_t_values = X_pred_full[valid_params_t] @ params[valid_params_t] if valid_params_t else 0.0

        decomp_df = data_for_pred.copy()
        decomp_df['F_a'] = F_a_values.values if isinstance(F_a_values, pd.Series) else F_a_values
        decomp_df['G_v'] = G_v_values.values if isinstance(G_v_values, pd.Series) else G_v_values
        decomp_df['H_t'] = H_t_values.values if isinstance(H_t_values, pd.Series) else H_t_values

        self.initial_decomp['F_a'] = decomp_df.groupby(self.age_col, observed=True)['F_a'].mean().sort_index()
        self.initial_decomp['G_v'] = decomp_df.groupby(self.vintage_col, observed=True)['G_v'].mean().sort_index()
        self.initial_decomp['H_t'] = decomp_df.groupby(self.time_col, observed=True)['H_t'].mean().sort_index()

        self.initial_decomp['H_prime_t'] = self.initial_decomp['H_t']
        F_a_df = self.initial_decomp['F_a'].reset_index()
        G_v_df = self.initial_decomp['G_v'].reset_index()
        alpha1_prime = get_linear_trend(F_a_df['F_a'], F_a_df[self.age_col])
        beta1_prime = get_linear_trend(G_v_df['G_v'], G_v_df[self.vintage_col])
        self.initial_decomp['alpha1_prime'] = alpha1_prime
        self.initial_decomp['beta1_prime'] = beta1_prime
        self.initial_decomp['F_nonlinear'] = self.initial_decomp['F_a'] - alpha1_prime * self.initial_decomp['F_a'].index
        self.initial_decomp['G_nonlinear'] = self.initial_decomp['G_v'] - beta1_prime * self.initial_decomp['G_v'].index
        self.initial_decomp['H_nonlinear'] = self.initial_decomp['H_t']


    def _fit_macro(self):
        """
        Step 2: Model the time component using macroeconomic variables.
        
        This step is critical for stress testing as it:
        1. Explains historical default patterns using economic factors
        2. Enables forecasting under different economic scenarios
        3. Separates economic impact from vintage quality effects
        
        The method:
        - Takes the time component (H) from initial decomposition
        - Regresses it against transformed macro variables
        - Builds a model: H'(t) ~ intercept + time_trend + UR + HPI + Credit_Score
        - Handles cases with insufficient data through fallback mechanisms
        
        Macro variables typically have these relationships with defaults:
        - Unemployment (UR): Positive coefficient (higher unemployment → higher defaults)
        - HPI: Negative coefficient (falling house prices → higher defaults)
        - Credit Scores: Negative coefficient (lower scores → higher defaults)
        
        The time_trend_col captures any remaining trend not explained by macro variables.
        """
        # Get the nonlinear time component from initial decomposition
        h_prime_t_series = self.initial_decomp.get('H_nonlinear', None)
        if h_prime_t_series is None or h_prime_t_series.empty:
             raise ValueError("Initial decomposition did not produce 'H_nonlinear' component.")

        # Ensure consistent rounding of time values for reliable merges
        h_prime_t_series.index = np.round(h_prime_t_series.index, self.round_precision)
        h_prime_df = h_prime_t_series.reset_index()
        
        # Set up macro data for merge
        macro_long_indexed = self.macro_data_long.set_index(self.time_col)

        # Ensure the H_nonlinear column is properly named
        if 'H_nonlinear' not in h_prime_df.columns:
             value_col = h_prime_df.columns.difference([self.time_col])
             if len(value_col) == 1: 
                h_prime_df = h_prime_df.rename(columns={value_col[0]: 'H_nonlinear'})
             else: 
                raise ValueError(f"Could not identify value column in h_prime_df. Cols: {h_prime_df.columns}")
        
        # Merge time component with macro variables
        # This aligns macro conditions with each time point in the portfolio
        regression_data_merged = pd.merge(
            h_prime_df,
            macro_long_indexed[['ur_transformed', 'hpi_transformed', 
                              'credit_score_transformed', 'time_trend_col']],
            left_on=self.time_col,
            right_index=True,
            how='left'
        )
        
        # Print diagnostic information about the merge result
        print(f"\n--- Merge Diagnostics ---")
        print(f"Portfolio time points range: {h_prime_df[self.time_col].min()} to {h_prime_df[self.time_col].max()}")
        print(f"Macro time points range: {macro_long_indexed.index.min()} to {macro_long_indexed.index.max()}")
        print(f"NaN counts after merge:\n{regression_data_merged.isna().sum().to_string()}")
        
        # Prepare data for regression
        regression_data_merged = regression_data_merged.rename(columns={'H_nonlinear': 'h_prime_target'})
        regression_data = regression_data_merged.dropna()

        # Check for sufficient observations to fit the full model
        num_parameters = 4  # Intercept + 3 macro variables
        if regression_data.shape[0] <= num_parameters:
             print("\n--- Insufficient Observations for Full Macro Model ---")
             print(f"Shape *before* dropna: {regression_data_merged.shape}")
             print(f"NaN counts *before* dropna:\n{regression_data_merged.isna().sum().to_string()}")
             print(f"Shape *after* dropna: {regression_data.shape}")
             
             # FALLBACK STRATEGY 1: Simplified model with filled NaN values
             print("\nAttempting fallback approach with simplified model...")
             
             # Keep rows with valid target and time_trend, fill NaNs in macro vars with zeros
             simplified_data = regression_data_merged.dropna(subset=['h_prime_target', 'time_trend_col'])
             simplified_data = simplified_data.fillna({
                 'ur_transformed': 0, 
                 'hpi_transformed': 0, 
                 'credit_score_transformed': 0
             })
             
             # FALLBACK STRATEGY 2: Create a synthetic trend model if still insufficient data
             if simplified_data.shape[0] <= 2:
                 print("\nInsufficient data even for simplified approach.")
                 print("Creating synthetic trend based on time component...")
                 
                 # Create a very basic linear trend model using available H values
                 h_t_df = h_prime_df.copy()
                 
                 # Ensure we have required columns
                 if 'H_nonlinear' not in h_t_df.columns:
                     h_t_df.rename(columns={h_t_df.columns[1]: 'H_nonlinear'}, inplace=True)
                 
                 # Fit a simple linear trend to H_nonlinear values
                 X = sm.add_constant(h_t_df[self.time_col])
                 y = h_t_df['H_nonlinear']
                 simple_trend_model = sm.OLS(y, X).fit()
                 
                 print(f"Synthetic trend coefficients: {simple_trend_model.params.to_dict()}")
                 
                 # Use these parameters as our macro model
                 self.macro_fit_results = simple_trend_model
                 self.macro_coeffs = pd.Series({
                     'Intercept': simple_trend_model.params['const'],
                     'time_trend_col': simple_trend_model.params[self.time_col],
                     'ur_transformed': 0.0,  # No macro effects in synthetic model
                     'hpi_transformed': 0.0,
                     'credit_score_transformed': 0.0
                 })
                 
                 # Store fit values
                 self.initial_decomp['E_fit_short'] = pd.Series(
                     simple_trend_model.predict(X), index=h_t_df[self.time_col])
                 print("Successfully created synthetic trend model as fallback.")
                 return
             
             # If we have enough data for simplified approach, use time trend only
             simplified_formula = "h_prime_target ~ 1 + time_trend_col"
             try:
                 y_simple, X_simple = dmatrices(simplified_formula, simplified_data, return_type='dataframe')
                 simple_model = sm.OLS(y_simple, X_simple)
                 simple_results = simple_model.fit()
                 
                 print("Successfully fit simplified model with just time trend.")
                 self.macro_fit_results = simple_results
                 self.macro_coeffs = pd.Series({
                     'Intercept': simple_results.params['Intercept'],
                     'time_trend_col': simple_results.params['time_trend_col'],
                     'ur_transformed': 0.0,  # Add dummy coefficients 
                     'hpi_transformed': 0.0,
                     'credit_score_transformed': 0.0
                 })
                 
                 fit_indices = simplified_data[self.time_col]
                 self.initial_decomp['E_fit_short'] = pd.Series(
                     simple_results.predict(X_simple), index=fit_indices)
                 return
                 
             except Exception as e:
                 print(f"Simplified model also failed: {e}")
                 print("Falling back to synthetic trend model...")
                 
                 # Same fallback as above
                 h_t_df = h_prime_df.copy()
                 if 'H_nonlinear' not in h_t_df.columns:
                     h_t_df.rename(columns={h_t_df.columns[1]: 'H_nonlinear'}, inplace=True)
                 
                 X = sm.add_constant(h_t_df[self.time_col])
                 y = h_t_df['H_nonlinear']
                 simple_trend_model = sm.OLS(y, X).fit()
                 
                 self.macro_fit_results = simple_trend_model
                 self.macro_coeffs = pd.Series({
                     'Intercept': simple_trend_model.params['const'],
                     'time_trend_col': simple_trend_model.params[self.time_col],
                     'ur_transformed': 0.0,
                     'hpi_transformed': 0.0,
                     'credit_score_transformed': 0.0
                 })
                 
                 self.initial_decomp['E_fit_short'] = pd.Series(
                     simple_trend_model.predict(X), index=h_t_df[self.time_col])
                 print("Successfully created synthetic trend model as fallback.")
                 return

        # STANDARD CASE: Fit full model with all macro variables
        # Full formula including credit scores
        formula_macro = "h_prime_target ~ 1 + time_trend_col + ur_transformed + hpi_transformed + credit_score_transformed"
        try:
            y_macro, X_macro = dmatrices(formula_macro, regression_data, return_type='dataframe')
        except Exception as e:
             print(f"Error during dmatrices call in _fit_macro: {e}")
             raise

        # Fit the model and store results
        macro_model = sm.OLS(y_macro, X_macro)
        self.macro_fit_results = macro_model.fit()
        self.macro_coeffs = self.macro_fit_results.params

        # Print model summary for analysis
        print("\n--- Macro Fit Results (Short Period) ---")
        print(self.macro_fit_results.summary())

        # Store predicted values for plotting
        fit_indices = regression_data[self.time_col]
        self.initial_decomp['E_fit_short'] = pd.Series(
            self.macro_fit_results.predict(X_macro), 
            index=fit_indices
        )


    def _calculate_long_term_trend(self):
        """Step 3: Extrapolate macro fit and find long-term trend eta_long."""
        if self.macro_coeffs is None: raise ValueError("Macro model must be fit first.")

        required_cols = ['time_trend_col', 'ur_transformed', 'hpi_transformed', 'credit_score_transformed']
        # Use the rounded time_col from macro_data_long
        predictors_long_raw = self.macro_data_long[required_cols + [self.time_col]].copy()
        predictors_long_raw = predictors_long_raw.dropna(subset=required_cols)
        if predictors_long_raw.empty: raise ValueError("No valid predictor data in long macro dataset.")

        formula_long_pred = "~ 1 + time_trend_col + ur_transformed + hpi_transformed + credit_score_transformed"
        try:
             predictors_long = dmatrix(formula_long_pred, predictors_long_raw, return_type='dataframe')
        except Exception as e:
             print(f"Error during dmatrix call in _calculate_long_term_trend: {e}"); raise


        try:
            coef_cols = self.macro_coeffs.index
            missing_cols = set(coef_cols) - set(predictors_long.columns)
            if missing_cols: raise ValueError(f"Missing columns in prediction data: {missing_cols}")
            predictors_long = predictors_long[coef_cols] # Align columns
        except Exception as e: raise ValueError(f"Error aligning prediction columns: {e}")

        E_fit_long_values = predictors_long.values @ self.macro_coeffs.values
        # Index using the rounded time_col from predictors_long_raw
        E_fit_long = pd.Series(E_fit_long_values, index=predictors_long_raw[self.time_col], name='E_fit_long')

        trend_calc_data_x = predictors_long_raw[self.time_col] # Already rounded
        trend_calc_data_y = E_fit_long

        if len(trend_calc_data_x) < 2:
             print("Warning: Less than 2 valid points for long-term trend calculation.")
             self.long_term_trend_eta = 0.0
        else:
            self.long_term_trend_eta = get_linear_trend(trend_calc_data_y, trend_calc_data_x)

        print(f"\nCalculated Long-Term Trend (eta_long): {self.long_term_trend_eta:.6f}")

        # Reindex using the full rounded time_col from macro_data_long
        self.initial_decomp['E_fit_long'] = E_fit_long.reindex(self.macro_data_long[self.time_col])
        
        if len(trend_calc_data_x) >= 2:
            intercept_long_trend = E_fit_long.mean() - self.long_term_trend_eta * trend_calc_data_x.mean()
        else: 
            intercept_long_trend = E_fit_long.mean() if not E_fit_long.empty else 0.0
            
        # Create trend line with the same index as E_fit_long to avoid index mismatch in plotting
        # This ensures we only calculate the trend line for points where E_fit_long is valid
        valid_indices = self.initial_decomp['E_fit_long'].dropna().index
        if len(valid_indices) > 0:
            self.initial_decomp['E_fit_long_trendline'] = pd.Series(
                intercept_long_trend + self.long_term_trend_eta * valid_indices,
                index=valid_indices
            )
        else:
            # Fallback to using all time points if E_fit_long has no valid values
            self.initial_decomp['E_fit_long_trendline'] = pd.Series(
                intercept_long_trend + self.long_term_trend_eta * self.macro_data_long[self.time_col],
                index=self.macro_data_long[self.time_col]
            )


    def _adjust_components(self):
        """Step 4 & 5: Adjust trends based on eta_long."""
        if self.long_term_trend_eta is None: raise ValueError("Long term trend must be calculated first.")
        eta_long = self.long_term_trend_eta
        alpha1_prime = self.initial_decomp['alpha1_prime']
        beta1_prime = self.initial_decomp['beta1_prime']
        alpha1_final = alpha1_prime + eta_long
        beta1_final = beta1_prime + eta_long

        self.final_components['F_a'] = self.initial_decomp['F_nonlinear'] + alpha1_final * self.initial_decomp['F_nonlinear'].index
        self.final_components['G_v'] = self.initial_decomp['G_nonlinear'] + beta1_final * self.initial_decomp['G_nonlinear'].index
        self.final_components['H_t'] = self.initial_decomp['H_nonlinear'] - eta_long * self.initial_decomp['H_nonlinear'].index
        self.final_components['F_nonlinear'] = self.initial_decomp['F_nonlinear']
        self.final_components['G_nonlinear'] = self.initial_decomp['G_nonlinear']
        self.final_components['H_nonlinear_final'] = self.final_components['H_t']


    def fit(self):
        """Runs the full fitting procedure."""
        print("Step 1: Performing Initial Decomposition...")
        self._initial_decomposition()
        print("Step 2: Fitting Macroeconomic Model to H'(t)...")
        self._fit_macro()
        print("Step 3: Calculating Long-Term Trend of Macro Fit...")
        self._calculate_long_term_trend()
        print("Step 4/5: Adjusting Component Trends...")
        self._adjust_components()
        print("Fitting Complete.")

    def predict(self, data):
        """Predicts log_odds on new data using final components."""
        if not self.final_components: raise ValueError("Model must be fitted before prediction.")
        df = data.copy()
        # Ensure prediction data time cols are rounded
        df[self.time_col] = df[self.time_col].round(self.round_precision)
        df[self.vintage_col] = df[self.vintage_col].round(self.round_precision)

        f_a = np.interp(df[self.age_col], self.final_components['F_a'].index, self.final_components['F_a'].values, left=self.final_components['F_a'].iloc[0], right=self.final_components['F_a'].iloc[-1])
        g_v = np.interp(df[self.vintage_col], self.final_components['G_v'].index, self.final_components['G_v'].values, left=self.final_components['G_v'].iloc[0], right=self.final_components['G_v'].iloc[-1])
        h_t = np.interp(df[self.time_col], self.final_components['H_t'].index, self.final_components['H_t'].values, left=self.final_components['H_t'].iloc[0], right=self.final_components['H_t'].iloc[-1])
        predictions = self.model_intercept + f_a + g_v + h_t
        return predictions

    # --- Plotting functions remain largely the same ---
    def plot_components(self, plot_initial=True):
        """Plots the initial and final estimated components."""
        if not self.final_components: print("Model not fitted yet."); return
        num_plots = 3; fig, axes = plt.subplots(num_plots, 1, figsize=(8, num_plots * 3.5))
        fig.suptitle('APC Model Components', fontsize=14)
        # Age
        if plot_initial and 'F_a' in self.initial_decomp: axes[0].plot(self.initial_decomp['F_a'].index, self.initial_decomp['F_a'].values, label='Initial F(a)', linestyle='--', alpha=0.7)
        axes[0].plot(self.final_components['F_a'].index, self.final_components['F_a'].values, label='Final F(a)'); axes[0].set_xlabel(f'Age ({self.age_col})'); axes[0].set_ylabel('Log-Odds Contribution'); axes[0].set_title('Age Component'); axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.6)
        # Vintage
        if plot_initial and 'G_v' in self.initial_decomp: axes[1].plot(self.initial_decomp['G_v'].index, self.initial_decomp['G_v'].values, label='Initial G(v)', linestyle='--', alpha=0.7)
        axes[1].plot(self.final_components['G_v'].index, self.final_components['G_v'].values, label='Final G(v)'); axes[1].set_xlabel(f'Vintage ({self.vintage_col})'); axes[1].set_ylabel('Log-Odds Contribution'); axes[1].set_title('Vintage Component'); axes[1].legend(); axes[1].grid(True, linestyle=':', alpha=0.6)
        # Time
        if plot_initial and 'H_t' in self.initial_decomp: axes[2].plot(self.initial_decomp['H_t'].index, self.initial_decomp['H_t'].values, label='Initial H(t)', linestyle='--', alpha=0.7)
        axes[2].plot(self.final_components['H_t'].index, self.final_components['H_t'].values, label='Final H(t)'); axes[2].set_xlabel(f'Time ({self.time_col})'); axes[2].set_ylabel('Log-Odds Contribution'); axes[2].set_title('Time (Environmental) Component'); axes[2].legend(); axes[2].grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show()

    def plot_macro_fit(self):
        """Plots the macro fit over the long term, similar to Fig 7."""
        if 'E_fit_long' not in self.initial_decomp or self.initial_decomp['E_fit_long'].isna().all(): 
            print("Macro fit not performed or long-term extrapolation missing/invalid.")
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        valid_long_fit = self.initial_decomp['E_fit_long'].dropna()
        
        if not valid_long_fit.empty:
            ax.plot(valid_long_fit.index, valid_long_fit.values, label='E_fit (Long Period)', linestyle='-', alpha=0.8)
            
            # Get only trend line values for indices that exist in valid_long_fit.index
            # This fixes issues where valid_long_fit has indices not present in trend line
            common_indices = valid_long_fit.index.intersection(self.initial_decomp['E_fit_long_trendline'].index)
            
            if len(common_indices) > 0:
                # Plot trend line using common indices
                valid_trend_line = self.initial_decomp['E_fit_long_trendline'].loc[common_indices]
                ax.plot(valid_trend_line.index, valid_trend_line.values, 
                        label=f'Long-Term Trend (Slope={self.long_term_trend_eta:.4f})', 
                        linestyle='--', color='red')
            else:
                # If no common indices, create a trend line manually for the visible range
                x_values = valid_long_fit.index
                intercept_trend = valid_long_fit.mean() - self.long_term_trend_eta * x_values.mean()
                trend_values = intercept_trend + self.long_term_trend_eta * x_values
                ax.plot(x_values, trend_values, 
                        label=f'Long-Term Trend (Slope={self.long_term_trend_eta:.4f})', 
                        linestyle='--', color='red')
                print("Warning: Used manual trend calculation for plot due to index mismatch")
        else: 
            print("Warning: No valid long-term fit points to plot.")
            
        if 'E_fit_short' in self.initial_decomp and not self.initial_decomp['E_fit_short'].empty:
            time_short = self.initial_decomp['E_fit_short'].index
            ax.plot(time_short, self.initial_decomp['E_fit_short'].values, 
                    label='E_fit (Short - In Sample)', linewidth=2, color='black', marker='.', markersize=4)
            
            min_short_time = time_short.min() 
            max_short_time = time_short.max()
            ax.axvline(min_short_time, color='gray', linestyle=':', label='Portfolio Data Range')
            ax.axvline(max_short_time, color='gray', linestyle=':')
        else: 
            print("Warning: Short fit data ('E_fit_short') not found for plotting.")
            
        ax.set_xlabel(f'Time ({self.time_col})')
        ax.set_ylabel('Log-Odds Contribution (Macro Fit)')
        ax.set_title('Macroeconomic Fit Extrapolation and Long-Term Trend')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Simulate Data with optimized parameters to reduce NaN issues
    portfolio_df, macro_short, macro_long = simulate_data(
        num_vintages=20,        # Increased from 10
        obs_per_vintage=60,     # Keep same length per vintage
        start_year=2007,
        round_precision=5,
        macro_history_years=3   # Reduced history (only need enough for lag/window)
    )

    print("Portfolio Data Sample:")
    print(portfolio_df.head())
    print(f"Portfolio Data Shape: {portfolio_df.shape}")
    print(f"Unique Time Points in Portfolio: {portfolio_df['time_numeric'].nunique()}")

    print("\nLong Macro Data Sample:")
    print(macro_long.head())
    print(f"Long Macro Data Shape: {macro_long.shape}")
    print(f"Unique Time Points in Long Macro: {macro_long['time_numeric'].nunique()}")
    
    print("\nMacro Data Columns:") 
    print(macro_long.columns.tolist())
    print("\nCredit Score Preview:")
    print(macro_long[['time_numeric', 'avg_credit_score']].head(10))

    # 2. Initialize and Fit the Model with minimal lag/window parameters
    try:
        tester = APCStressTester(
            portfolio_data=portfolio_df,
            macro_data_long=macro_long,
            spline_df_a=6, spline_df_v=5, spline_df_t=6,
            ur_lag=1,           # Minimal lag
            ur_window=2,        # Minimal window
            hpi_lag=1,          # Minimal lag
            hpi_window=2,       # Minimal window
            credit_score_lag=2, # Moderate lag for credit scores
            credit_score_window=3, # Short window for smoothing
            round_precision=5
        )
        tester.fit()

        # 3. Plot Results
        tester.plot_macro_fit()
        tester.plot_components(plot_initial=True)

        # 4. Predict on portfolio data (example)
        predictions = tester.predict(portfolio_df)
        portfolio_df['log_odds_predicted'] = predictions

        # Compare actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(portfolio_df['log_odds_default'], portfolio_df['log_odds_predicted'], alpha=0.3, s=10)
        min_val = min(portfolio_df['log_odds_default'].min(), portfolio_df['log_odds_predicted'].min())
        max_val = max(portfolio_df['log_odds_default'].max(), portfolio_df['log_odds_predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        plt.xlabel('Actual Log-Odds Default')
        plt.ylabel('Predicted Log-Odds Default')
        plt.title('Model Fit: Actual vs. Predicted (In-Sample)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    except ValueError as e:
        print(f"\nERROR during model fitting or prediction: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()