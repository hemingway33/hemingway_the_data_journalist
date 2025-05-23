import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
from lifelines.datasets import load_dd
import matplotlib.pyplot as plt # Added for plotting

# --- Constants for data generation and hypothetical profiles ---
CENTER_AGE = 40
CENTER_INCOME = 50000
CENTER_LOAN_AMOUNT = 25000
CENTER_LOAN_DURATION = 24 # This one is more for the data generation logic

def generate_credit_default_data_time_varying(n_subjects=500, max_time=60):
    """
    Generates synthetic credit default data with time-varying covariates.
    Events are now stochastically generated based on covariate values in each segment.
    """
    np.random.seed(42)

    # True coefficients and baseline hazard are defined INSIDE the function, 
    # as they are specific to the data generation process itself.
    # If they were needed externally, they might be returned or made global too.
    TRUE_BETA_AGE = 0.04
    TRUE_BETA_INCOME = -0.00003
    TRUE_BETA_LOAN_AMOUNT = 0.00002
    TRUE_BETA_LOAN_DURATION = 0.02 
    BASE_LOG_HAZARD = np.log(0.005) 

    data_rows = []

    for i in range(n_subjects):
        # Static features for the subject
        age = np.random.randint(22, 65)
        income = np.random.normal(50000, 15000)
        if income < 5000: income = 5000 # Ensure minimum income
        loan_amount_orig = np.random.uniform(5000, 50000)

        current_time = 0
        is_event_occurred_for_subject = False
        
        while current_time < max_time and not is_event_occurred_for_subject:
            segment_duration = np.random.randint(1, 13) # Duration of this observation window
            start_time_segment = current_time
            stop_time_segment = min(current_time + segment_duration, max_time)
            actual_segment_duration = stop_time_segment - start_time_segment

            if actual_segment_duration <= 0: # Should not happen if max_time > 0
                break

            # loan_duration_months is the duration of the loan *at the end of this segment*
            current_loan_duration_at_stop = stop_time_segment 

            # Calculate log-hazard for this segment based on covariates
            log_hazard_for_segment = (
                BASE_LOG_HAZARD +
                TRUE_BETA_AGE * (age - CENTER_AGE) +
                TRUE_BETA_INCOME * (income - CENTER_INCOME) +
                TRUE_BETA_LOAN_AMOUNT * (loan_amount_orig - CENTER_LOAN_AMOUNT) +
                TRUE_BETA_LOAN_DURATION * (current_loan_duration_at_stop - CENTER_LOAN_DURATION)
            )
            hazard_rate_for_segment = np.exp(log_hazard_for_segment)

            # Probability of event occurring in this segment
            prob_event_in_segment = 1 - np.exp(-hazard_rate_for_segment * actual_segment_duration)
            
            event_status_for_segment = 0
            final_stop_time_for_row = stop_time_segment

            if np.random.rand() < prob_event_in_segment:
                event_status_for_segment = 1
                is_event_occurred_for_subject = True
                # Simulate more precise event time within the segment for realism, if desired.
                # For simplicity here, if event occurs in segment, it's recorded at stop_time_segment.
                # A more precise time could be: start_time_segment + np.random.exponential(1/hazard_rate_for_segment)
                # and then final_stop_time_for_row = min(precise_event_time, stop_time_segment)
                # We'll keep it simple: event at stop_time_segment of the current interval.

            data_rows.append({
                'id': i,
                'age': age,
                'income': round(income, 2),
                'loan_amount': round(loan_amount_orig, 2),
                'start_time': round(start_time_segment, 2),
                'stop_time': round(final_stop_time_for_row, 2),
                'loan_duration_months': round(current_loan_duration_at_stop,2),
                'event': event_status_for_segment
            })
            
            current_time = final_stop_time_for_row
            # Loop breaks if is_event_occurred_for_subject is True or current_time reaches max_time

    df = pd.DataFrame(data_rows)
    df = df[df['start_time'] < df['stop_time']] # Ensure valid intervals
    df['income'] = df['income'].clip(lower=0)
    return df

# Generate the data
df_credit = generate_credit_default_data_time_varying(n_subjects=200, max_time=72) # 72 months = 6 years
print("Sample of generated data:")
print(df_credit.head())
print(f"\nNumber of rows: {len(df_credit)}")
print(f"Number of unique subjects: {df_credit['id'].nunique()}")
print(f"\nEvent distribution:\n{df_credit[df_credit['event'] == 1].groupby('id').first()['event'].value_counts(normalize=True)}")


# Fit the Cox Proportional Hazards model with time-varying covariates
ctv = CoxTimeVaryingFitter(penalizer=0.01) # Added a small penalizer for stability

# Define covariates
# 'loan_duration_months' is our time-varying covariate
# 'age', 'income', 'loan_amount' are static (but need to be in the df for each interval)
covariates = ['age', 'income', 'loan_amount', 'loan_duration_months']

print("\nFitting CoxTimeVaryingFitter model...")
ctv.fit(df_credit,
        id_col='id',
        event_col='event',
        start_col='start_time',
        stop_col='stop_time',
        show_progress=True,
        strata=None) # No stratification for now

print("\nModel Summary:")
ctv.print_summary(decimals=4)

# --- Prediction ---
# Prediction with CoxTimeVaryingFitter is a bit more involved than with standard CoxPH
# as you need to provide the future covariate values.

# Example: Predict survival function for a specific individual or a new dataset
# For prediction, you'd typically have a DataFrame with the covariate history and future values.

# Let's take a subset of our data to predict on (e.g., first few subjects, their last known state)
# To make meaningful predictions, you'd construct a new dataframe representing
# the future time intervals for subjects you want to predict for.

# For demonstration, let's predict the survival probability for the existing subjects
# at their respective 'stop_time's, given their history.
# Note: This is more for illustration of the predict_survival_function method.
# In a real scenario, you'd predict for future time points.

print("\n--- Example: Predicting Survival Function ---")
# Let's try to predict for a hypothetical scenario for one subject (e.g., id=0)
# We need to create a dataframe representing the intervals for which we want predictions.
subject_id_to_predict = df_credit['id'].unique()[0]
subject_data = df_credit[df_credit['id'] == subject_id_to_predict].copy()

# If we want to predict future survival, we need to define future time intervals
# and the expected values of covariates in those intervals.
# For instance, predict survival for the next 12 months for subject_id_to_predict,
# assuming their static covariates remain the same, and loan_duration_months increases.

if not subject_data.empty:
    last_observation = subject_data.iloc[[-1]]
    
    future_predictions_df_rows = []
    current_stop_time = last_observation['stop_time'].iloc[0]
    
    # Let's predict for 3 future 1-month intervals
    for i in range(1, 4):
        future_start = current_stop_time + (i-1)
        future_stop = current_stop_time + i
        
        if future_start >= future_stop: continue # skip if interval is zero

        future_predictions_df_rows.append({
            'id': subject_id_to_predict,
            'age': last_observation['age'].iloc[0], # Static
            'income': last_observation['income'].iloc[0], # Static
            'loan_amount': last_observation['loan_amount'].iloc[0], # Static
            'start_time': future_start,
            'stop_time': future_stop,
            'loan_duration_months': last_observation['loan_duration_months'].iloc[0] + i # Time-varying
            # 'event' column is not needed for prediction data
        })
    
    future_df_for_prediction = pd.DataFrame(future_predictions_df_rows)

    if not future_df_for_prediction.empty:
        print(f"\nPredicting future survival for subject {subject_id_to_predict} with hypothetical future data:")
        print(future_df_for_prediction)
        
        try:
            baseline_hazard_df = ctv.baseline_cumulative_hazard_
            covariate_names = ctv.params_.index.tolist() # Get covariate names from the fitted model

            # Helper function to get H0(t) from the baseline cumulative hazard DataFrame
            def get_H0_t(t, bch_df):
                if bch_df.empty or t < bch_df.index.min():
                    return 0.0
                
                relevant_bch_entries = bch_df[bch_df.index <= t]
                
                if relevant_bch_entries.empty:
                    # This state implies t is < the first time in bch_df.index,
                    # already handled by t < bch_df.index.min()
                    return 0.0 
                
                # Assuming the cumulative hazard value is in the first column
                return relevant_bch_entries.iloc[-1, 0]

            subject_cumulative_hazard_path_values = []
            current_total_cumulative_hazard = 0.0
            predicted_survival_probabilities = []

            for idx, interval_data_row in future_df_for_prediction.iterrows():
                # Extract covariates for the current interval
                # Ensure X_interval_df has the correct columns in the order expected by the model
                X_interval_df = pd.DataFrame([interval_data_row[covariate_names]], columns=covariate_names)
                
                # Calculate partial hazard (risk score: exp(beta*X)) for this interval's covariates
                # predict_partial_hazard returns a Pandas Series
                partial_hazard_for_interval = ctv.predict_partial_hazard(X_interval_df).iloc[0]

                interval_start_time = interval_data_row['start_time']
                interval_stop_time = interval_data_row['stop_time']

                # Get baseline cumulative hazard at the start and end of the interval
                H0_at_interval_start = get_H0_t(interval_start_time, baseline_hazard_df)
                H0_at_interval_stop = get_H0_t(interval_stop_time, baseline_hazard_df)
                
                # Increment in baseline cumulative hazard over this interval
                delta_H0_for_interval = H0_at_interval_stop - H0_at_interval_start
                
                # Cumulative hazard contribution from this specific interval
                hazard_contribution_this_interval = partial_hazard_for_interval * delta_H0_for_interval
                
                # Update total cumulative hazard up to the end of this interval
                current_total_cumulative_hazard += hazard_contribution_this_interval
                subject_cumulative_hazard_path_values.append(current_total_cumulative_hazard)
                
                # Calculate survival probability at the end of this interval: S(t) = exp(-H(t))
                survival_prob_at_interval_stop = np.exp(-current_total_cumulative_hazard)
                predicted_survival_probabilities.append(survival_prob_at_interval_stop)

            print(f"\nPredicted CUMULATIVE HAZARD for future intervals of subject {subject_id_to_predict} (path-based calculation):")
            cumulative_hazard_series = pd.Series(subject_cumulative_hazard_path_values, index=future_df_for_prediction['stop_time'])
            print(cumulative_hazard_series)

            print(f"\nPredicted SURVIVAL PROBABILITY for future intervals of subject {subject_id_to_predict} (path-based calculation):")
            survival_prob_series = pd.Series(predicted_survival_probabilities, index=future_df_for_prediction['stop_time'])
            print(survival_prob_series)

        except Exception as e:
            print(f"Error during prediction: {e}")
            print("Prediction with CoxTimeVaryingFitter can be complex. The `predict_survival_function` expects data X, and for each row in X, it computes S(t | X_i).")
            print("For a sequence of time-varying covariates for one individual, you'd typically calculate cumulative hazard over the path.")
else:
    print(f"\nSubject {subject_id_to_predict} not found or has no data.")

print("\n--- Finished ---")

# --- Plotting Survival Curves ---
print("\n--- Plotting Survival Curves ---")

# 1. Plot Baseline Survival Curve
plt.figure(figsize=(12, 7))
try:
    ctv.baseline_survival_.plot(label="Baseline Survival S0(t)")
    plt.title("Baseline Survival Curve")
    plt.ylabel("Survival Probability")
    plt.xlabel("Time (months)")
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Could not plot baseline survival: {e}")


# 2. Plot Predicted Survival Curves for Hypothetical Profiles

def predict_survival_for_profile(profile_df, ctv_model, covariate_names_from_model):
    """Calculates survival probabilities for a given profile path."""
    baseline_hazard_df = ctv_model.baseline_cumulative_hazard_
    
    # Helper function to get H0(t) from the baseline cumulative hazard DataFrame
    # (Copied from earlier prediction logic, ensure it's consistent or refactor into a shared utility)
    def get_H0_t(t, bch_df):
        if bch_df.empty or t < bch_df.index.min():
            return 0.0
        relevant_bch_entries = bch_df[bch_df.index <= t]
        if relevant_bch_entries.empty:
            return 0.0 
        return relevant_bch_entries.iloc[-1, 0]

    profile_cumulative_hazard_path = []
    current_total_cumulative_hazard = 0.0
    profile_survival_probabilities = []
    profile_stop_times = []

    for _, interval_data_row in profile_df.iterrows():
        X_interval_df = pd.DataFrame([interval_data_row[covariate_names_from_model]], columns=covariate_names_from_model)
        partial_hazard_for_interval = ctv_model.predict_partial_hazard(X_interval_df).iloc[0]

        interval_start_time = interval_data_row['start_time']
        interval_stop_time = interval_data_row['stop_time']

        H0_at_interval_start = get_H0_t(interval_start_time, baseline_hazard_df)
        H0_at_interval_stop = get_H0_t(interval_stop_time, baseline_hazard_df)
        delta_H0_for_interval = H0_at_interval_stop - H0_at_interval_start
        
        hazard_contribution_this_interval = partial_hazard_for_interval * delta_H0_for_interval
        current_total_cumulative_hazard += hazard_contribution_this_interval
        
        profile_cumulative_hazard_path.append(current_total_cumulative_hazard)
        profile_survival_probabilities.append(np.exp(-current_total_cumulative_hazard))
        profile_stop_times.append(interval_stop_time)
        
    return pd.Series(profile_survival_probabilities, index=profile_stop_times)

# Define time horizon for plotting hypothetical profiles
prediction_horizon_months = 36
monthly_intervals = range(1, prediction_horizon_months + 1)

# Profile 1: Average Risk
profile1_data = []
for t_stop in monthly_intervals:
    profile1_data.append({
        'id': 'profile1',
        'age': CENTER_AGE,
        'income': CENTER_INCOME,
        'loan_amount': CENTER_LOAN_AMOUNT,
        'start_time': t_stop - 1, # Assuming 1-month intervals from time 0
        'stop_time': t_stop,
        'loan_duration_months': t_stop # loan_duration_months = current time in this simple path
    })
profile1_df = pd.DataFrame(profile1_data)

# Profile 2: Higher Risk
profile2_data = []
for t_stop in monthly_intervals:
    profile2_data.append({
        'id': 'profile2',
        'age': CENTER_AGE + 15,       # Older
        'income': CENTER_INCOME * 0.7, # Lower income
        'loan_amount': CENTER_LOAN_AMOUNT * 1.5, # Higher loan amount
        'start_time': t_stop - 1,
        'stop_time': t_stop,
        'loan_duration_months': t_stop 
    })
profile2_df = pd.DataFrame(profile2_data)


plt.figure(figsize=(12, 7))
try:
    # Get covariate names from the fitted model (used in predict_survival_for_profile)
    fitted_covariate_names = ctv.params_.index.tolist()

    # Predict and plot for Profile 1
    survival_profile1 = predict_survival_for_profile(profile1_df, ctv, fitted_covariate_names)
    plt.plot(survival_profile1.index, survival_profile1.values, label="Avg Risk Profile (Age:40, Inc:50k, Loan:25k)")

    # Predict and plot for Profile 2
    survival_profile2 = predict_survival_for_profile(profile2_df, ctv, fitted_covariate_names)
    plt.plot(survival_profile2.index, survival_profile2.values, label="Higher Risk Profile (Age:55, Inc:35k, Loan:37.5k)")
    
    # Optionally, plot baseline survival again for comparison if not shown on separate plot
    # ctv.baseline_survival_.plot(label="Baseline Survival S0(t)", linestyle='--')

    plt.title("Predicted Survival Curves for Hypothetical Profiles")
    plt.ylabel("Survival Probability")
    plt.xlabel("Time (months) into Future Path")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05) # Ensure y-axis is from 0 to 1
    plt.show()

except Exception as e:
    print(f"Could not plot hypothetical profiles: {e}")
    print("Ensure the model is fitted and ctv.params_ is available.")
