"""
Vintage-Level Full Model for Loan Portfolio Stress Testing
========================================================

This module implements a vintage-level full model that supports simultaneous 
Age-Period-Cohort (APC) decomposition and multi-factor regression for credit
risk stress testing, based on "Solutions to specification errors in stress testing models".

Key features:
- Simultaneous estimation of age, period, and cohort effects
- Multi-factor regression with macroeconomic variables
- Proper identification through constraints
- Forecasting capabilities for stress scenario analysis

References:
- Breeden, J. L. (2016). "Solutions to specification errors in stress testing models."
  Journal of the Operational Research Society, 67(11), 1398-1409.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix, dmatrices
import matplotlib.pyplot as plt
from scipy import optimize
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Helper Functions ---

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

class VintageFullModel:
    """
    Vintage-level full model that supports simultaneous APC decomposition and multi-factor regression
    
    This model addresses the identification problem in APC analysis through constraints
    and provides an integrated approach to estimating base effects and macro impacts.
    
    Parameters:
    -----------
    portfolio_data : pandas.DataFrame
        Vintage-level loan performance data with age, vintage, time, and target columns
    macro_data : pandas.DataFrame
        Macroeconomic data with various indicators
    age_col : str, default='age_months'
        Column name for loan age
    vintage_col : str, default='vintage_numeric'
        Column name for loan vintage indicator
    time_col : str, default='time_numeric'
        Column name for observation time
    target_col : str, default='log_odds_default'
        Column name for the target variable (should be log-odds)
    spline_df_a : int, default=5
        Degrees of freedom for age spline
    spline_df_v : int, default=5
        Degrees of freedom for vintage spline
    spline_df_t : int, default=5
        Degrees of freedom for time spline
    round_precision : int, default=5
        Decimal precision for rounding time values
    """
    
    def __init__(self, portfolio_data, macro_data,
                 age_col='age_months', vintage_col='vintage_numeric', time_col='time_numeric',
                 target_col='log_odds_default', 
                 spline_df_a=5, spline_df_v=5, spline_df_t=5,
                 round_precision=5):
        
        self.portfolio_data = portfolio_data.copy()
        self.macro_data = macro_data.copy()
        self.age_col = age_col
        self.vintage_col = vintage_col
        self.time_col = time_col
        self.target_col = target_col
        self.spline_df_a = spline_df_a
        self.spline_df_v = spline_df_v
        self.spline_df_t = spline_df_t
        self.round_precision = round_precision
        
        # Initialize containers for model components
        self.model = None
        self.results = None
        self.effects = None
        self.macro_coefs = None
        self.macro_formula = None
        self.X_apc = None
        self.X_macro = None
        
        # For storage of component values
        self.age_effect = None
        self.vintage_effect = None
        self.time_effect = None
        self.intercept = None
        
    def prepare_data(self):
        """Prepares the data for modeling by creating splines and merging with macro data."""
        df = self.portfolio_data.copy()
        
        # Round time variables to ensure precise merging
        df[self.age_col] = np.round(df[self.age_col], self.round_precision)
        df[self.vintage_col] = np.round(df[self.vintage_col], self.round_precision)
        df[self.time_col] = np.round(df[self.time_col], self.round_precision)
        
        # Create APC design matrices using natural splines
        age_spline = dmatrix(f"cr(x, df={self.spline_df_a})", 
                             {"x": df[self.age_col].values}, 
                             return_type='dataframe')
        age_spline.columns = [f'age_{i}' for i in range(age_spline.shape[1])]
        
        vintage_spline = dmatrix(f"cr(x, df={self.spline_df_v})", 
                                 {"x": df[self.vintage_col].values}, 
                                 return_type='dataframe')
        vintage_spline.columns = [f'vintage_{i}' for i in range(vintage_spline.shape[1])]
        
        time_spline = dmatrix(f"cr(x, df={self.spline_df_t})", 
                              {"x": df[self.time_col].values}, 
                              return_type='dataframe')
        time_spline.columns = [f'time_{i}' for i in range(time_spline.shape[1])]
        
        # Combine splines into one design matrix
        X_apc = pd.concat([age_spline, vintage_spline, time_spline], axis=1)
        
        # Check if the time column exists in the macro data, if not add it
        macro_data = self.macro_data.copy()
        if self.time_col not in macro_data.columns:
            # Try to find a suitable time column
            possible_time_cols = ['time', 'date', 'period', 'month', 'quarter', 'year']
            found_col = None
            
            for col in possible_time_cols:
                if col in macro_data.columns:
                    found_col = col
                    break
                
            if found_col:
                # Rename the found column to match time_col
                macro_data = macro_data.rename(columns={found_col: self.time_col})
            else:
                # If we can't find a time column, print a warning and create a simple index-based one
                print(f"Warning: Time column '{self.time_col}' not found in macro data. Creating a numeric time index.")
                unique_times = sorted(df[self.time_col].unique())
                if len(unique_times) <= len(macro_data):
                    # If we have enough macro data rows, use the actual time values
                    macro_data[self.time_col] = unique_times[:len(macro_data)]
                else:
                    # Otherwise just use row indices as time points
                    macro_data[self.time_col] = np.arange(len(macro_data))
                
        # Round the time column in macro data to match portfolio data precision
        macro_data[self.time_col] = np.round(macro_data[self.time_col], self.round_precision)
        
        # Merge with macro data
        try:
            df_with_macro = df.merge(macro_data, on=self.time_col, how='left')
            
            # Check if merge produced NaN values and warn user
            nan_rows = df_with_macro.isna().any(axis=1).sum()
            if nan_rows > 0:
                print(f"Warning: Merge with macro data produced {nan_rows} rows with missing values.")
                print(f"Unique time values in portfolio data: {sorted(df[self.time_col].unique())}")
                print(f"Unique time values in macro data: {sorted(macro_data[self.time_col].unique())}")
        except Exception as e:
            print(f"Error merging data: {str(e)}")
            print("Proceeding without macro data...")
            df_with_macro = df.copy()
        
        # Store the prepared data
        self.prepared_data = df_with_macro
        self.X_apc = X_apc
        
        return df_with_macro
        
    def add_macro_variables(self, macro_formula):
        """
        Adds macroeconomic variables to the model based on a specified formula.
        
        Parameters:
        -----------
        macro_formula : str
            A patsy formula string specifying macro variables and transformations
            Example: "unemployment + hpi_change"
        """
        self.macro_formula = macro_formula
        
        if not hasattr(self, 'prepared_data'):
            self.prepare_data()
            
        # Create design matrix for macro variables
        X_macro = dmatrix(macro_formula, data=self.prepared_data, return_type='dataframe')
        
        # Store for later use
        self.X_macro = X_macro
        
    def fit(self, constraints='zero_sum', fit_intercept=True):
        """
        Fits the full vintage model with APC and macro effects.
        
        Parameters:
        -----------
        constraints : str, default='zero_sum'
            Type of constraints to apply for identification:
            - 'zero_sum': Sum of effects equal zero
            - 'first_zero': First coefficient in each effect set to zero
            - 'last_zero': Last coefficient in each effect set to zero
        fit_intercept : bool, default=True
            Whether to include an intercept in the model
        """
        if not hasattr(self, 'X_macro') or self.X_macro is None:
            raise ValueError("No macro variables added. Call add_macro_variables() first.")
            
        # Combine APC and macro design matrices
        X_full = pd.concat([self.X_apc, self.X_macro], axis=1)
        
        # Extract target variable
        y = self.prepared_data[self.target_col]
        
        # Remove missing values
        mask = ~(X_full.isna().any(axis=1) | y.isna())
        X_full_clean = X_full[mask]
        y_clean = y[mask]
        
        # Apply constraints for identification
        constraint_dict = self._build_constraint_matrix(X_full_clean.columns, constraints)
        
        # Fit model
        model = sm.OLS(y_clean, X_full_clean)
        
        if constraints == 'zero_sum':
            # For zero sum, we need to use add_constraint
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools.tools import add_constant
            
            # Get constraint matrix and vector
            R, r = constraint_dict
            
            # Create design matrix and fit
            X_array = np.asarray(X_full_clean)
            y_array = np.asarray(y_clean)
            
            # Solve the constrained least squares problem manually
            # This is equivalent to: min ||Xb - y||^2 subject to Rb = r
            from scipy import linalg
            
            # Method 1: Using Lagrange multipliers
            X_t = X_array.T
            XX = X_t.dot(X_array)
            Xy = X_t.dot(y_array)
            
            # Stack for the KKT system
            R = np.array(R)  # Convert list to array
            
            # Create the block matrix for KKT system
            top_block = np.hstack([XX, R.T])
            bottom_block = np.hstack([R, np.zeros((len(R), len(R)))])
            A = np.vstack([top_block, bottom_block])
            
            # RHS vector
            b = np.concatenate([Xy, r])
            
            # Solve the system
            try:
                sol = linalg.solve(A, b)
                params = sol[:XX.shape[0]]  # First part contains coefficients
                
                # Create a results object
                results = model.fit()
                results.params = pd.Series(params, index=X_full_clean.columns)
                
            except np.linalg.LinAlgError:
                # If singular matrix, fall back to unconstrained fit with warning
                print("Warning: Constraint matrix is singular. Falling back to unconstrained fit.")
                results = model.fit()
        else:
            # For first_zero or last_zero, we can use parameter restrictions
            results = model.fit()
            
            # Apply zero constraints by setting parameters
            if isinstance(constraint_dict, dict) and constraint_dict:
                for param, value in constraint_dict.items():
                    if param in results.params.index:
                        # Get parameter index
                        param_idx = results.params.index.get_loc(param)
                        
                        # Refit without the constrained variable
                        X_reduced = X_full_clean.copy()
                        X_reduced = X_reduced.drop(columns=[param])
                        
                        # Adjust dependent variable
                        if value != 0:
                            y_adjusted = y_clean - value * X_full_clean[param]
                        else:
                            y_adjusted = y_clean
                        
                        # Fit reduced model
                        model_reduced = sm.OLS(y_adjusted, X_reduced)
                        results_reduced = model_reduced.fit()
                        
                        # Insert the constrained value
                        params = results_reduced.params.copy()
                        new_params = pd.Series(index=results.params.index)
                        
                        for i, col in enumerate(results.params.index):
                            if col == param:
                                new_params[col] = value
                            elif col in params.index:
                                new_params[col] = params[col]
                            else:
                                new_params[col] = 0
                        
                        # Update results with new parameters
                        results.params = new_params
        
        # Store results
        self.model = model
        self.results = results
        
        # Extract effects
        self._extract_effects()
        
        return self
    
    def _build_constraint_matrix(self, columns, constraints_type):
        """Builds the constraint matrix for model identification."""
        age_cols = [col for col in columns if col.startswith('age_')]
        vintage_cols = [col for col in columns if col.startswith('vintage_')]
        time_cols = [col for col in columns if col.startswith('time_')]
        
        if constraints_type == 'zero_sum':
            # Create zero-sum constraints for each effect
            R = []
            
            if len(age_cols) > 1:
                age_constraint = np.zeros(len(columns))
                for i, col in enumerate(columns):
                    if col in age_cols:
                        age_constraint[i] = 1
                R.append(age_constraint)
                
            if len(vintage_cols) > 1:
                vintage_constraint = np.zeros(len(columns))
                for i, col in enumerate(columns):
                    if col in vintage_cols:
                        vintage_constraint[i] = 1
                R.append(vintage_constraint)
                
            if len(time_cols) > 1:
                time_constraint = np.zeros(len(columns))
                for i, col in enumerate(columns):
                    if col in time_cols:
                        time_constraint[i] = 1
                R.append(time_constraint)
                
            r = np.zeros(len(R))
            
            return (R, r)
            
        elif constraints_type == 'first_zero':
            # Set first coefficient in each group to zero
            constraints = {}
            if age_cols:
                constraints[age_cols[0]] = 0
            if vintage_cols:
                constraints[vintage_cols[0]] = 0
            if time_cols:
                constraints[time_cols[0]] = 0
            return constraints
            
        elif constraints_type == 'last_zero':
            # Set last coefficient in each group to zero
            constraints = {}
            if age_cols:
                constraints[age_cols[-1]] = 0
            if vintage_cols:
                constraints[vintage_cols[-1]] = 0
            if time_cols:
                constraints[time_cols[-1]] = 0
            return constraints
        
        return None
    
    def _extract_effects(self):
        """Extracts and organizes the APC effects from the model results."""
        # Initialize storage for effects
        age_indices = self.prepared_data[self.age_col].unique()
        vintage_indices = self.prepared_data[self.vintage_col].unique()
        time_indices = self.prepared_data[self.time_col].unique()
        
        age_effect = pd.Series(0.0, index=age_indices)
        vintage_effect = pd.Series(0.0, index=vintage_indices)
        time_effect = pd.Series(0.0, index=time_indices)
        
        params = self.results.params
        
        # Extract age effect coefficients
        age_cols = [col for col in params.index if col.startswith('age_')]
        if age_cols:
            age_basis = dmatrix(f"cr(x, df={self.spline_df_a})", 
                               {"x": age_effect.index.values}, 
                               return_type='dataframe')
            age_basis.columns = [f'age_{i}' for i in range(age_basis.shape[1])]
            age_effect_values = age_basis.dot(params[age_cols]).values
            age_effect = pd.Series(age_effect_values, index=age_indices)
        
        # Extract vintage effect coefficients
        vintage_cols = [col for col in params.index if col.startswith('vintage_')]
        if vintage_cols:
            vintage_basis = dmatrix(f"cr(x, df={self.spline_df_v})", 
                                   {"x": vintage_effect.index.values}, 
                                   return_type='dataframe')
            vintage_basis.columns = [f'vintage_{i}' for i in range(vintage_basis.shape[1])]
            vintage_effect_values = vintage_basis.dot(params[vintage_cols]).values
            vintage_effect = pd.Series(vintage_effect_values, index=vintage_indices)
        
        # Extract time effect coefficients
        time_cols = [col for col in params.index if col.startswith('time_')]
        if time_cols:
            time_basis = dmatrix(f"cr(x, df={self.spline_df_t})", 
                               {"x": time_effect.index.values}, 
                               return_type='dataframe')
            time_basis.columns = [f'time_{i}' for i in range(time_basis.shape[1])]
            time_effect_values = time_basis.dot(params[time_cols]).values
            time_effect = pd.Series(time_effect_values, index=time_indices)
        
        # Store as pandas Series
        self.age_effect = age_effect
        self.vintage_effect = vintage_effect
        self.time_effect = time_effect
        
        # Extract macro coefficients
        macro_cols = [col for col in params.index if not (
            col.startswith('age_') or col.startswith('vintage_') or col.startswith('time_'))]
        self.macro_coefs = params[macro_cols]
        
        # Extract intercept if present
        if 'Intercept' in params:
            self.intercept = params['Intercept']
        else:
            self.intercept = 0.0
            
        # Store combined effects dictionary
        self.effects = {
            'age': self.age_effect,
            'vintage': self.vintage_effect,
            'time': self.time_effect,
            'macro': self.macro_coefs,
            'intercept': self.intercept
        }
        
    def predict(self, newdata=None):
        """
        Generates predictions using the fitted model.
        
        Parameters:
        -----------
        newdata : pandas.DataFrame, optional
            New data for prediction. If None, predictions are made on the training data.
            
        Returns:
        --------
        pandas.Series
            Predicted values
        """
        if newdata is None:
            return self.results.fittedvalues
        
        # Prepare new data similarly to training data
        df = newdata.copy()
        
        # Round time variables to ensure precise merging
        df[self.age_col] = np.round(df[self.age_col], self.round_precision)
        df[self.vintage_col] = np.round(df[self.vintage_col], self.round_precision)
        df[self.time_col] = np.round(df[self.time_col], self.round_precision)
        
        # Merge with macro data
        df_with_macro = df.merge(self.macro_data, on=self.time_col, how='left')
        
        # Create APC design matrices
        age_spline = dmatrix(f"cr(x, df={self.spline_df_a})", 
                             {"x": df[self.age_col].values}, 
                             return_type='dataframe')
        age_spline.columns = [f'age_{i}' for i in range(age_spline.shape[1])]
        
        vintage_spline = dmatrix(f"cr(x, df={self.spline_df_v})", 
                                 {"x": df[self.vintage_col].values}, 
                                 return_type='dataframe')
        vintage_spline.columns = [f'vintage_{i}' for i in range(vintage_spline.shape[1])]
        
        time_spline = dmatrix(f"cr(x, df={self.spline_df_t})", 
                              {"x": df[self.time_col].values}, 
                              return_type='dataframe')
        time_spline.columns = [f'time_{i}' for i in range(time_spline.shape[1])]
        
        # Combine splines
        X_apc = pd.concat([age_spline, vintage_spline, time_spline], axis=1)
        
        # Create macro design matrix
        X_macro = dmatrix(self.macro_formula, data=df_with_macro, return_type='dataframe')
        
        # Combine all predictors
        X_full = pd.concat([X_apc, X_macro], axis=1)
        
        # Ensure columns match the fitted model
        missing_cols = set(self.results.params.index) - set(X_full.columns)
        extra_cols = set(X_full.columns) - set(self.results.params.index)
        
        if missing_cols:
            for col in missing_cols:
                X_full[col] = 0
                
        X_full = X_full[self.results.params.index]
        
        # Generate predictions
        predictions = self.results.predict(X_full)
        
        return predictions
    
    def forecast_scenarios(self, scenario_data, periods=12):
        """
        Forecasts under different economic scenarios.
        
        Parameters:
        -----------
        scenario_data : dict of pandas.DataFrame
            Dictionary with scenario names as keys and dataframes of macro forecasts as values
        periods : int, default=12
            Number of forecast periods
            
        Returns:
        --------
        dict
            Dictionary with scenario names as keys and forecast DataFrames as values
        """
        forecasts = {}
        
        # Get the last observed data point
        last_time = self.prepared_data[self.time_col].max()
        last_vintage = self.prepared_data[self.vintage_col].max()
        
        for scenario_name, scenario_df in scenario_data.items():
            # Create forecast grid for each vintage
            forecast_data = []
            
            for vintage in self.prepared_data[self.vintage_col].unique():
                # Only include vintages that still have active loans
                if vintage > last_vintage - self.prepared_data[self.age_col].max():
                    for t in range(1, periods + 1):
                        time_point = last_time + t
                        age = time_point - vintage
                        
                        # Only include valid ages
                        if age >= 0 and age <= self.prepared_data[self.age_col].max():
                            forecast_data.append({
                                self.age_col: age,
                                self.vintage_col: vintage,
                                self.time_col: time_point
                            })
            
            # Convert to DataFrame
            forecast_df = pd.DataFrame(forecast_data)
            
            # Ensure scenario_df has the time column before merging
            scenario_df_copy = scenario_df.copy()
            
            # Check if the time column exists in the scenario data
            if self.time_col not in scenario_df_copy.columns:
                # Try to find a suitable time column
                possible_time_cols = ['time', 'date', 'period', 'month', 'quarter', 'year']
                found_col = None
                
                for col in possible_time_cols:
                    if col in scenario_df_copy.columns:
                        found_col = col
                        break
                        
                if found_col:
                    # Rename the found column to match time_col
                    scenario_df_copy = scenario_df_copy.rename(columns={found_col: self.time_col})
                else:
                    # If we can't find a time column, use the forecast time points
                    print(f"Warning: Time column '{self.time_col}' not found in {scenario_name} scenario data.")
                    # Generate a time sequence that matches forecast_df's time points
                    unique_times = sorted(forecast_df[self.time_col].unique())
                    
                    # Create a dataframe with just the time column
                    time_df = pd.DataFrame({self.time_col: unique_times})
                    
                    # If scenario_df is empty, just use time_df
                    if len(scenario_df_copy) == 0:
                        scenario_df_copy = time_df
                    else:
                        # Otherwise, try to align the existing data with times
                        if len(unique_times) >= len(scenario_df_copy):
                            # We have more forecast periods than scenario rows - align at the start
                            scenario_df_copy[self.time_col] = unique_times[:len(scenario_df_copy)]
                        else:
                            # We have fewer forecast periods - truncate the scenario data
                            scenario_df_copy = scenario_df_copy.iloc[:len(unique_times)]
                            scenario_df_copy[self.time_col] = unique_times
            
            # Ensure time column is properly rounded in both dataframes
            forecast_df[self.time_col] = np.round(forecast_df[self.time_col], self.round_precision)
            scenario_df_copy[self.time_col] = np.round(scenario_df_copy[self.time_col], self.round_precision)
            
            try:
                # Merge with scenario data
                forecast_with_macro = forecast_df.merge(scenario_df_copy, on=self.time_col, how='left')
                
                # Check if merge produced NaN values and warn user
                nan_cols = forecast_with_macro.columns[forecast_with_macro.isna().any()]
                if len(nan_cols) > 0:
                    print(f"Warning: Merge with {scenario_name} scenario produced missing values for columns: {list(nan_cols)}")
                    print(f"Forecast periods: {sorted(forecast_df[self.time_col].unique())}")
                    print(f"Scenario periods: {sorted(scenario_df_copy[self.time_col].unique())}")
                    
                    # For missing macro vars, use the last available value
                    for col in nan_cols:
                        if col != self.time_col and col != self.age_col and col != self.vintage_col:
                            if not forecast_with_macro[col].isna().all():
                                # Fill NaN with the last valid value
                                forecast_with_macro[col] = forecast_with_macro[col].fillna(method='ffill')
                            else:
                                # If all values are NaN, fill with zero
                                print(f"Warning: All values for '{col}' are missing. Filling with 0.")
                                forecast_with_macro[col] = 0
                
                # Make predictions
                forecast_with_macro[f'forecast_{scenario_name}'] = self.predict(forecast_with_macro)
            except Exception as e:
                print(f"Error in forecasting {scenario_name} scenario: {str(e)}")
                print("Creating forecast with only base APC effects...")
                
                # Fall back to using just the forecast grid with base effects
                forecast_with_macro = forecast_df.copy()
                
                # Generate predictions using only APC effects, no macro variables
                age_vals = forecast_df[self.age_col].values
                vintage_vals = forecast_df[self.vintage_col].values
        return forecasts
    
    def plot_components(self, figsize=(15, 12)):
        """
        Plots the decomposed components of the model.
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 12)
            Figure size as (width, height)
        """
        if not hasattr(self, 'effects'):
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Age effect
        axes[0].plot(self.age_effect.index, self.age_effect.values)
        axes[0].set_title('Age Effect')
        axes[0].set_xlabel('Loan Age (months)')
        axes[0].set_ylabel('Log-Odds Contribution')
        axes[0].grid(True)
        
        # Vintage effect
        axes[1].plot(self.vintage_effect.index, self.vintage_effect.values)
        axes[1].set_title('Vintage Effect')
        axes[1].set_xlabel('Vintage Time')
        axes[1].set_ylabel('Log-Odds Contribution')
        axes[1].grid(True)
        
        # Time effect
        axes[2].plot(self.time_effect.index, self.time_effect.values)
        axes[2].set_title('Time Effect')
        axes[2].set_xlabel('Observation Time')
        axes[2].set_ylabel('Log-Odds Contribution')
        axes[2].grid(True)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_macro_impact(self, figsize=(15, 6)):
        """
        Plots the impact of macroeconomic variables on the predicted values.
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 6)
            Figure size as (width, height)
        """
        if not hasattr(self, 'macro_coefs'):
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Create bar plot of macro coefficients
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = self.macro_coefs.index
        values = self.macro_coefs.values
        
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title('Macroeconomic Variable Impacts')
        ax.set_ylabel('Coefficient Value (Log-Odds Impact)')
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        return fig, ax
    
    def summary(self):
        """
        Returns a summary of the model results.
        
        Returns:
        --------
        statsmodels.iolib.summary.Summary
            Summary of the model results
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        return self.results.summary()


# Example usage function
def run_example():
    """
    Runs an example of the VintageFullModel on simulated data.
    """
    # Import simulation function from vintage_apc.py
    from vintage_apc import simulate_data
    
    # Simulate data
    portfolio_df, macro_short, macro_long = simulate_data(
        num_vintages=10, 
        obs_per_vintage=40, 
        start_year=2005
    )
    
    # Convert default rates to log-odds
    portfolio_df['default_rate'] = inv_log_odds(portfolio_df['log_odds_default'])
    
    # Initialize the model
    model = VintageFullModel(
        portfolio_data=portfolio_df,
        macro_data=macro_short,
        age_col='age_months',
        vintage_col='vintage_numeric',
        time_col='time_numeric',
        target_col='log_odds_default',
        spline_df_a=5,
        spline_df_v=5,
        spline_df_t=5
    )
    
    # Prepare data
    model.prepare_data()
    
    # Add macro variables
    model.add_macro_variables("unemployment + I(unemployment - unemployment.shift(3)) + I(hpi/hpi.shift(12) - 1)")
    
    # Fit the model
    model.fit(constraints='zero_sum')
    
    # Print summary
    print(model.summary())
    
    # Plot components
    model.plot_components()
    plt.show()
    
    # Plot macro impacts
    model.plot_macro_impact()
    plt.show()
    
    # Create forecast scenarios
    base_scenario = macro_short.copy()
    stress_scenario = macro_short.copy()
    
    # Extend scenarios to future periods
    last_time = macro_short['time_numeric'].max()
    future_times = pd.DataFrame({
        'time_numeric': np.arange(last_time + 1, last_time + 13)
    })
    
    # Base scenario: continue recent trends
    for col in ['unemployment', 'hpi', 'avg_credit_score']:
        if col in base_scenario.columns:
            last_values = base_scenario[col].tail(6).values
            trend = (last_values[-1] - last_values[0]) / 5
            future_values = np.array([last_values[-1] + trend * (i+1) for i in range(12)])
            
            base_scenario_future = future_times.copy()
            base_scenario_future[col] = future_values
            
            base_scenario = pd.concat([base_scenario, base_scenario_future])
    
    # Stress scenario: unemployment up, HPI down, credit scores down
    stress_scenario_future = future_times.copy()
    
    if 'unemployment' in stress_scenario.columns:
        last_unemp = stress_scenario['unemployment'].iloc[-1]
        stress_scenario_future['unemployment'] = [
            last_unemp + i * 0.3 for i in range(12)
        ]
    
    if 'hpi' in stress_scenario.columns:
        last_hpi = stress_scenario['hpi'].iloc[-1]
        stress_scenario_future['hpi'] = [
            last_hpi * (1 - i * 0.01) for i in range(12)
        ]
    
    if 'avg_credit_score' in stress_scenario.columns:
        last_score = stress_scenario['avg_credit_score'].iloc[-1]
        stress_scenario_future['avg_credit_score'] = [
            last_score - i * 2 for i in range(12)
        ]
    
    stress_scenario = pd.concat([stress_scenario, stress_scenario_future])
    
    # Create scenario dictionary
    scenarios = {
        'base': base_scenario,
        'stress': stress_scenario
    }
    
    # Generate forecasts
    forecasts = model.forecast_scenarios(scenarios, periods=12)
    
    # Plot forecasts
    plt.figure(figsize=(12, 6))
    
    for scenario_name, forecast_df in forecasts.items():
        # Group by time and average the forecasts
        agg_forecast = forecast_df.groupby(model.time_col)[f'forecast_{scenario_name}'].mean()
        plt.plot(agg_forecast.index, inv_log_odds(agg_forecast.values) * 100, 
                 label=f'{scenario_name.capitalize()} Scenario')
    
    # Add historical data
    historical = portfolio_df.groupby(model.time_col)['default_rate'].mean() * 100
    plt.plot(historical.index, historical.values, 'k-', label='Historical')
    
    plt.xlabel('Time')
    plt.ylabel('Default Rate (%)')
    plt.title('Forecast Scenarios')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, forecasts


if __name__ == "__main__":
    run_example()
