import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Simulation Parameters ---
N_LOANS = 10000
AVG_BALANCE = 100000
INTEREST_RATE_ANNUAL = 0.10 # 10% APR
INTEREST_RATE_MONTHLY = INTEREST_RATE_ANNUAL / 12
N_BINS = 8
LOANS_PER_BIN = N_LOANS // N_BINS # Assuming equal distribution
HIGH_RISK_BINS = [6, 7] # Bin indices (0-7), so Bins 7 and 8

# Predicted Default (PD) rates per bin (0-indexed)
# Let's assume higher bins have higher PD
PD_RATES = np.array([0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.20, 0.35])

# Strategy 1 Parameters
RECOVERY_RATES_STRAT1 = np.array([0.1, 0.2, 0.3, 0.5]) # LGD = 1 - Recovery Rate

# Strategy 2 Parameters
DEFAULT_REDUCTION_FACTORS = np.array([0.2, 0.4, 0.6, 0.8]) # e.g., 0.2 means PD reduces by 20%
EXTENSION_MONTHS = 24
# Assume a standard recovery rate for defaults occurring in non-recalled/non-extended loans or during extension
STANDARD_RECOVERY_RATE = 0.3

# Funding Cost
FUNDING_COST_RATE_ANNUAL = 0.03 # 3% annual cost of funds

# --- Calculations ---
TOTAL_PORTFOLIO_VALUE = N_LOANS * AVG_BALANCE

# Original Loan Term: 12 months, balloon payment
# Interest paid monthly, principal at the end
MONTHLY_INTEREST_PAYMENT = AVG_BALANCE * INTEREST_RATE_MONTHLY
TOTAL_INTEREST_ORIGINAL = MONTHLY_INTEREST_PAYMENT * 11 # Interest for first 11 months

def calculate_baseline_profit_loss():
    """Calculates the expected profit/loss and absolute loss for the portfolio with no intervention, including funding cost."""
    total_expected_profit = 0
    total_expected_loss_on_principal = 0
    results_per_bin = []

    for i in range(N_BINS):
        n_loans_in_bin = LOANS_PER_BIN
        pd_rate = PD_RATES[i]
        
        # Expected outcomes for this bin
        expected_defaults = n_loans_in_bin * pd_rate
        expected_non_defaults = n_loans_in_bin * (1 - pd_rate)

        # Profit from non-defaults: Full interest for 11 months
        profit_non_default = expected_non_defaults * TOTAL_INTEREST_ORIGINAL
        
        # Loss from defaults: Principal lost minus recovery.
        loss_principal_default = expected_defaults * AVG_BALANCE * (1 - STANDARD_RECOVERY_RATE)
        
        total_expected_profit += profit_non_default
        total_expected_loss_on_principal += loss_principal_default
        results_per_bin.append({
            'Bin': i,
            'PD Rate': pd_rate,
            'Loans': n_loans_in_bin,
            'Expected Defaults': expected_defaults,
            'Expected Non-Defaults': expected_non_defaults,
            'Bin Profit': profit_non_default,
            'Bin Principal Loss': loss_principal_default,
            'Bin Net (Gross)': profit_non_default - loss_principal_default
        })
        
    gross_net_profit_loss = total_expected_profit - total_expected_loss_on_principal
    
    # Funding Cost (1 year for all loans)
    funding_cost = TOTAL_PORTFOLIO_VALUE * FUNDING_COST_RATE_ANNUAL
    
    net_profit_loss_after_funding = gross_net_profit_loss - funding_cost
    annualized_rate = net_profit_loss_after_funding / TOTAL_PORTFOLIO_VALUE
    
    print(f"Baseline: Gross Net P&L: ${gross_net_profit_loss:,.2f}, Total Abs Loss: ${total_expected_loss_on_principal:,.2f}, Funding Cost: ${funding_cost:,.2f}, Net P&L After Funding: ${net_profit_loss_after_funding:,.2f}, Annualized Rate: {annualized_rate:.2%}")
    return net_profit_loss_after_funding, annualized_rate, total_expected_loss_on_principal, pd.DataFrame(results_per_bin)

def simulate_strategy_1(recovery_rate_recall):
    """Simulates Strategy 1: Recall loans, returning P&L and absolute loss."""
    total_expected_profit = 0
    total_expected_loss_on_principal = 0

    for i in range(N_BINS):
        n_loans_in_bin = LOANS_PER_BIN
        pd_rate = PD_RATES[i]
        is_high_risk = i in HIGH_RISK_BINS

        expected_defaults = n_loans_in_bin * pd_rate
        expected_non_defaults = n_loans_in_bin * (1 - pd_rate)
        
        if is_high_risk:
            # Strategy 1 Applied: Recall
            # Non-defaults pay off at month 12 -> Profit is interest earned
            profit_non_default = expected_non_defaults * TOTAL_INTEREST_ORIGINAL
            # Defaults occur at month 12 -> Loss is principal * (1 - recovery_rate_recall)
            loss_principal_default = expected_defaults * AVG_BALANCE * (1 - recovery_rate_recall)
        else:
            # Baseline logic for lower-risk bins
            profit_non_default = expected_non_defaults * TOTAL_INTEREST_ORIGINAL
            loss_principal_default = expected_defaults * AVG_BALANCE * (1 - STANDARD_RECOVERY_RATE)

        total_expected_profit += profit_non_default
        total_expected_loss_on_principal += loss_principal_default
        
    gross_net_profit_loss = total_expected_profit - total_expected_loss_on_principal
    
    # Funding Cost (1 year for all loans, as recall happens at expiry)
    funding_cost = TOTAL_PORTFOLIO_VALUE * FUNDING_COST_RATE_ANNUAL
    
    net_profit_loss_after_funding = gross_net_profit_loss - funding_cost
    annualized_rate = net_profit_loss_after_funding / TOTAL_PORTFOLIO_VALUE

    return net_profit_loss_after_funding, annualized_rate, total_expected_loss_on_principal

def calculate_amortization_payment(principal, monthly_rate, num_months):
    """Calculates the fixed monthly payment for an amortizing loan."""
    if monthly_rate == 0:
        return principal / num_months
    return principal * (monthly_rate * (1 + monthly_rate)**num_months) / ((1 + monthly_rate)**num_months - 1)

def calculate_outstanding_balance(principal, monthly_rate, num_months, payment_number):
    """Calculates the outstanding balance after a certain number of payments."""
    if monthly_rate == 0:
        return principal - (principal / num_months) * payment_number
    monthly_payment = calculate_amortization_payment(principal, monthly_rate, num_months)
    balance = principal * (1 + monthly_rate)**payment_number - \
              monthly_payment * (((1 + monthly_rate)**payment_number - 1) / monthly_rate)
    return max(0, balance) # Ensure balance doesn't go below zero due to floating point issues

def calculate_exponential_survival_probs(annual_pd, max_months):
    """Calculates monthly survival probabilities assuming an exponential distribution.

    Args:
        annual_pd: The annualized probability of default.
        max_months: The maximum number of months to calculate for.

    Returns:
        A list S where S[m] is the probability of surviving past month m. S[0]=1.
    """
    if annual_pd >= 1.0:
        # If annual PD is 100% or more, survival beyond month 0 is impossible.
        # Hazard rate lambda is effectively infinite.
        probs = [1.0] + [0.0] * max_months
        return probs
    if annual_pd <= 0:
        # If annual PD is 0 or less, survival is certain.
        return [1.0] * (max_months + 1)
        
    # Calculate the constant hazard rate lambda from annual PD
    # annual_pd = 1 - exp(-lambda * 1 year)
    hazard_lambda = -np.log(1 - annual_pd)
    
    # Calculate monthly survival probabilities S(m) = exp(-lambda * m / 12)
    months = np.arange(max_months + 1)
    survival_probs = np.exp(-hazard_lambda * months / 12.0)
    return survival_probs.tolist()

def simulate_strategy_2(default_reduction_factor):
    """Simulates Strategy 2: Extend loans, returning P&L, total return, avg duration, and absolute loss."""
    total_gross_pnl_low_risk = 0
    total_gross_pnl_high_risk = 0
    total_expected_loss_on_principal_low_risk = 0
    total_expected_loss_on_principal_high_risk = 0
    expected_duration_high_risk = 0 # To calculate average funding duration

    n_high_risk_loans = len(HIGH_RISK_BINS) * LOANS_PER_BIN
    n_low_risk_loans = N_LOANS - n_high_risk_loans
    value_high_risk = n_high_risk_loans * AVG_BALANCE
    value_low_risk = n_low_risk_loans * AVG_BALANCE

    # Pre-calculate for extended loans
    extended_monthly_payment = calculate_amortization_payment(AVG_BALANCE, INTEREST_RATE_MONTHLY, EXTENSION_MONTHS)
    total_interest_extension_if_survives = (extended_monthly_payment * EXTENSION_MONTHS) - AVG_BALANCE

    for i in range(N_BINS):
        n_loans_in_bin = LOANS_PER_BIN
        pd_rate_annual = PD_RATES[i]
        is_high_risk = i in HIGH_RISK_BINS

        if is_high_risk:
            # Strategy 2 Applied: Extension - Detailed Monthly Simulation using Survival Function
            reduced_pd_annual = pd_rate_annual * (1 - default_reduction_factor)
            reduced_pd_annual = max(0, min(reduced_pd_annual, 0.99999)) # Clamp PD to avoid log(0) or >=1 issues

            # Get monthly survival probabilities based on the reduced annual PD
            # S[m] = probability of surviving past month m
            survival_probs = calculate_exponential_survival_probs(reduced_pd_annual, EXTENSION_MONTHS)
            
            expected_pnl_per_loan = 0
            bin_expected_duration = 0
            expected_loss_per_loan = 0 # Track absolute loss separately

            for t in range(1, EXTENSION_MONTHS + 1):
                # Probability of defaulting *in* month t = S(t-1) - S(t)
                prob_default_in_month_t = survival_probs[t-1] - survival_probs[t]

                if prob_default_in_month_t > 1e-9: # Check if probability is non-negligible
                    # Calculate P&L if default occurs in month t
                    outstanding_balance_at_t = calculate_outstanding_balance(AVG_BALANCE, INTEREST_RATE_MONTHLY, EXTENSION_MONTHS, t)
                    # Consider payments made up to t-1
                    payments_made = extended_monthly_payment * (t - 1)
                    interest_paid_before_t = payments_made - (AVG_BALANCE - calculate_outstanding_balance(AVG_BALANCE, INTEREST_RATE_MONTHLY, EXTENSION_MONTHS, t-1)) if t > 1 else 0
                    
                    loss_on_principal_at_t = outstanding_balance_at_t * (1 - STANDARD_RECOVERY_RATE)
                    pnl_if_default_at_t = interest_paid_before_t - loss_on_principal_at_t

                    expected_pnl_per_loan += prob_default_in_month_t * pnl_if_default_at_t
                    bin_expected_duration += prob_default_in_month_t * (t / 12.0) # Add weighted duration in years
                    expected_loss_per_loan += prob_default_in_month_t * loss_on_principal_at_t # Accumulate expected principal loss

            # Add P&L for survival case
            # Prob of surviving all 24 months = S(EXTENSION_MONTHS)
            prob_survival_total = survival_probs[EXTENSION_MONTHS]
            if prob_survival_total > 1e-9:
                pnl_if_survives = total_interest_extension_if_survives
                expected_pnl_per_loan += prob_survival_total * pnl_if_survives
                bin_expected_duration += prob_survival_total * (EXTENSION_MONTHS / 12.0) # Add weighted duration for survivors

            total_gross_pnl_high_risk += n_loans_in_bin * expected_pnl_per_loan
            total_expected_loss_on_principal_high_risk += n_loans_in_bin * expected_loss_per_loan
            # Accumulate weighted average duration across high-risk bins
            expected_duration_high_risk += (n_loans_in_bin / n_high_risk_loans) * bin_expected_duration 

        else:
            # Baseline logic for lower-risk bins (1 year)
            expected_defaults = n_loans_in_bin * pd_rate_annual
            expected_non_defaults = n_loans_in_bin * (1 - pd_rate_annual)
            profit_non_default = expected_non_defaults * TOTAL_INTEREST_ORIGINAL
            loss_principal_default = expected_defaults * AVG_BALANCE * (1 - STANDARD_RECOVERY_RATE)
            total_gross_pnl_low_risk += (profit_non_default - loss_principal_default)
            total_expected_loss_on_principal_low_risk += loss_principal_default

    # Combine P&L and Loss from low and high risk bins
    gross_net_profit_loss = total_gross_pnl_low_risk + total_gross_pnl_high_risk
    total_expected_loss_on_principal = total_expected_loss_on_principal_low_risk + total_expected_loss_on_principal_high_risk
    
    # Refined Funding Cost:
    # Low-risk loans: 1 year cost
    # High-risk loans (extended): Use expected duration
    funding_cost_low_risk = value_low_risk * FUNDING_COST_RATE_ANNUAL * 1
    funding_cost_high_risk = value_high_risk * FUNDING_COST_RATE_ANNUAL * expected_duration_high_risk 
    total_funding_cost = funding_cost_low_risk + funding_cost_high_risk

    net_profit_loss_after_funding = gross_net_profit_loss - total_funding_cost
    
    # Still report total P&L after funding. Annualized rate is complex.
    # Calculate an average overall duration for context?
    avg_portfolio_duration = (n_low_risk_loans/N_LOANS * 1.0) + (n_high_risk_loans/N_LOANS * expected_duration_high_risk)
    # Simple overall return = PnL / Value
    total_return_overall = net_profit_loss_after_funding / TOTAL_PORTFOLIO_VALUE 

    # Return total PnL after funding, and the overall return rate (not annualized) and avg duration
    return net_profit_loss_after_funding, total_return_overall, avg_portfolio_duration, total_expected_loss_on_principal

# --- Main Execution ---
if __name__ == "__main__":
    baseline_net_af, baseline_rate, baseline_loss, baseline_details_df = calculate_baseline_profit_loss()
    # print("\nBaseline Details:")
    # print(baseline_details_df.to_string())

    strategy1_results = {}
    strategy1_rates = {}
    strategy1_losses = {}
    print("\nSimulating Strategy 1 (Recall)...")
    for recovery_rate in RECOVERY_RATES_STRAT1:
        net_pnl_af, ann_rate, total_loss = simulate_strategy_1(recovery_rate)
        strategy1_results[recovery_rate] = net_pnl_af
        strategy1_rates[recovery_rate] = ann_rate
        strategy1_losses[recovery_rate] = total_loss
        print(f"  Recovery Rate {recovery_rate:.1f}: Net P&L After Funding: ${net_pnl_af:,.2f}, Annualized Rate: {ann_rate:.2%}, Total Abs Loss: ${total_loss:,.2f}")

    # We will add Strategy 2 calculations here later
    strategy2_results = {}
    strategy2_total_rates = {}
    strategy2_avg_durations = {}
    strategy2_losses = {}
    print("\nSimulating Strategy 2 (Extension - Detailed)...")
    for reduction_factor in DEFAULT_REDUCTION_FACTORS:
        net_pnl_af, total_rate_overall, avg_duration, total_loss = simulate_strategy_2(reduction_factor)
        strategy2_results[reduction_factor] = net_pnl_af
        strategy2_total_rates[reduction_factor] = total_rate_overall
        strategy2_avg_durations[reduction_factor] = avg_duration
        strategy2_losses[reduction_factor] = total_loss
        print(f"  Reduction Factor {reduction_factor:.1f}: Net P&L After Funding: ${net_pnl_af:,.2f}, Overall Return: {total_rate_overall:.2%}, Avg Duration: {avg_duration:.2f} yrs, Total Abs Loss: ${total_loss:,.2f}")
    
    print("\nSimulation Analysis Complete.")

    # --- Visualization ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle('Loan Portfolio Exit Strategy Simulation Results (Net P&L After Funding Cost)', fontsize=16)

    # Plot Baseline on both
    ax1.axhline(baseline_net_af, color='grey', linestyle='--', linewidth=2, label=f'Baseline (${baseline_net_af:,.0f})')
    ax2.axhline(baseline_net_af, color='grey', linestyle='--', linewidth=2, label=f'Baseline (${baseline_net_af:,.0f})')

    # Strategy 1 Plot
    s1_rates_keys = list(strategy1_results.keys())
    s1_pnl_values = list(strategy1_results.values())
    ax1.plot(s1_rates_keys, s1_pnl_values, marker='o', linestyle='-', linewidth=2, markersize=8, color='skyblue', label='Strategy 1 (Recall)')
    ax1.set_xlabel('Recovery Rate (Strategy 1)', fontsize=12)
    ax1.set_ylabel('Net Profit/Loss After Funding Cost ($)', fontsize=12)
    ax1.set_title('Strategy 1 (Recall) vs. Baseline', fontsize=14)
    ax1.legend()
    ax1.grid(True)
    ax1.ticklabel_format(style='plain', axis='y') # Prevent scientific notation

    # Strategy 2 Plot
    s2_factors_keys = list(strategy2_results.keys())
    s2_pnl_values = list(strategy2_results.values())
    ax2.plot(s2_factors_keys, s2_pnl_values, marker='s', linestyle='-', linewidth=2, markersize=8, color='lightcoral', label='Strategy 2 (Extension)')
    ax2.set_xlabel('Default Reduction Factor (Strategy 2)', fontsize=12)
    ax2.set_title('Strategy 2 (Extension) vs. Baseline', fontsize=14)
    ax2.legend()
    ax2.grid(True)
    ax2.ticklabel_format(style='plain', axis='y') # Prevent scientific notation

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

    # --- Visualization for Absolute Loss ---
    fig_loss, (ax_loss1, ax_loss2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig_loss.suptitle('Loan Portfolio Exit Strategy Simulation Results (Total Expected Absolute Loss)', fontsize=16)

    # Plot Baseline Loss on both
    ax_loss1.axhline(baseline_loss, color='grey', linestyle='--', linewidth=2, label=f'Baseline Loss (${baseline_loss:,.0f})')
    ax_loss2.axhline(baseline_loss, color='grey', linestyle='--', linewidth=2, label=f'Baseline Loss (${baseline_loss:,.0f})')

    # Strategy 1 Loss Plot
    s1_loss_values = list(strategy1_losses.values())
    ax_loss1.plot(s1_rates_keys, s1_loss_values, marker='o', linestyle='-', linewidth=2, markersize=8, color='orange', label='Strategy 1 (Recall)')
    ax_loss1.set_xlabel('Recovery Rate (Strategy 1)', fontsize=12)
    ax_loss1.set_ylabel('Total Expected Absolute Loss ($)', fontsize=12)
    ax_loss1.set_title('Strategy 1 Absolute Loss vs. Baseline', fontsize=14)
    ax_loss1.legend()
    ax_loss1.grid(True)
    ax_loss1.ticklabel_format(style='plain', axis='y')

    # Strategy 2 Loss Plot
    s2_loss_values = list(strategy2_losses.values())
    ax_loss2.plot(s2_factors_keys, s2_loss_values, marker='s', linestyle='-', linewidth=2, markersize=8, color='red', label='Strategy 2 (Extension)')
    ax_loss2.set_xlabel('Default Reduction Factor (Strategy 2)', fontsize=12)
    ax_loss2.set_title('Strategy 2 Absolute Loss vs. Baseline', fontsize=14)
    ax_loss2.legend()
    ax_loss2.grid(True)
    ax_loss2.ticklabel_format(style='plain', axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Further Analysis Suggestion ---
    # To directly compare S1 and S2, we could create a heatmap or contour plot 
    # showing the difference (S2_pnl - S1_pnl) as a function of 
    # Recovery Rate (S1) and Reduction Factor (S2). This would require running S2
    # simulation across different *assumed* recovery rates if default occurs,
    # or running S1 across different assumptions for comparison.
    # For now, the two separate plots provide a clear view of each strategy's sensitivity.
    print("\nVisualization generated.")


# Example of how to calculate S2 PnL varying both reduction and recovery
# This was not explicitly requested but shows how a direct comparison could be built

def simulate_strategy_2_detailed(default_reduction_factor, recovery_rate_extension):
    """Simulates Strategy 2 allowing varying recovery rate for extension defaults.
       Needs update to match the detailed monthly simulation logic if used.
    """
    # ... (Implementation would need to mirror the main simulate_strategy_2 logic) ...
    pass # Placeholder - needs rewriting based on new logic

# You could then run this nestedly:
# s2_detailed_results = pd.DataFrame(index=DEFAULT_REDUCTION_FACTORS, columns=RECOVERY_RATES_STRAT1)
# for factor in DEFAULT_REDUCTION_FACTORS:
#     for recovery in RECOVERY_RATES_STRAT1:
#         s2_detailed_results.loc[factor, recovery] = simulate_strategy_2_detailed(factor, recovery)
# print("\nStrategy 2 PnL (Rows: Reduction Factor, Cols: Recovery Rate)")
# print(s2_detailed_results.to_string(float_format="{:,.0f}"))
