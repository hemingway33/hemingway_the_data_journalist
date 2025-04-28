import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Simulation Parameters ---
N_LOANS = 10000
AVG_BALANCE = 100000
INTEREST_RATE_ANNUAL = 0.12 # 12% APR
INTEREST_RATE_MONTHLY = INTEREST_RATE_ANNUAL / 12
N_BINS = 8
LOANS_PER_BIN = N_LOANS // N_BINS # Assuming equal distribution
HIGH_RISK_BINS = [6, 7] # Bin indices (0-7), so Bins 7 and 8

# Predicted Default (PD) rates per bin (0-indexed)
# Let's assume higher bins have higher PD
PD_RATES = np.array([0.02, 0.04, 0.06, 0.08, 0.12, 0.15, 0.20, 0.35])

# Strategy 1 Parameters
RECOVERY_RATES_STRAT1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) # LGD = 1 - Recovery Rate

# Strategy 2 & 3 Parameters
DEFAULT_REDUCTION_FACTORS = np.array([0.2, 0.3, 0.4, 0.5, 0.6]) # e.g., 0.2 means PD reduces by 20%
EXTENSION_MONTHS = 24
PRINCIPAL_REDUCTION_FACTOR = 0.10 # 10% haircut on principal for extended loans
REFUSAL_PROBABILITY = 0.05 # 5% probability customer refuses extension and defaults
EXTENSION_RATE_INCREASE = 0.05 # Absolute increase in annual rate for extended loans (e.g., 0.05 means +5%)

# Assume a standard recovery rate for defaults occurring in non-recalled/non-extended loans or during extension
STANDARD_RECOVERY_RATE = 0.3

# Funding Cost
FUNDING_COST_RATE_ANNUAL = 0.03 # 3% annual cost of funds

# --- Multi-Year Simulation Parameters ---
CHURN_RATE = 0.30 # 30% of active loans churn (pay off) each year
SIMULATION_YEARS = 3 # Simulate for 10 years
DISCOUNT_RATE = FUNDING_COST_RATE_ANNUAL # Use funding cost as discount rate for NPV

# --- Calculations ---
TOTAL_PORTFOLIO_VALUE = N_LOANS * AVG_BALANCE

# Original Loan Term: 12 months, balloon payment
# Interest paid monthly, principal at the end
MONTHLY_INTEREST_PAYMENT = AVG_BALANCE * INTEREST_RATE_MONTHLY
TOTAL_INTEREST_ORIGINAL = MONTHLY_INTEREST_PAYMENT * 11 # Interest for first 11 months

# We might need to represent the portfolio state more dynamically now.
# Let's use a dictionary or DataFrame later if needed.

# --- NPV Calculation Helper ---
def calculate_npv(cash_flows, rate):
    """Calculates the Net Present Value of a series of cash flows."""
    npv = 0
    if not cash_flows: # Handle empty cash flow list
        return 0
    cumulative_npv = []
    for year_index, cf in enumerate(cash_flows):
        # Ensure cf is treated as a number
        cf_numeric = float(cf) if cf is not None else 0.0
        npv += cf_numeric / ((1 + rate) ** (year_index + 1)) # Discounting year 1 CF by (1+r)^1, etc.
        cumulative_npv.append(npv) # Store NPV up to this year
    return npv, cumulative_npv # Return final NPV and yearly cumulative NPV

# --- Helper Functions (Amortization, Survival etc. - Keep as needed) ---
def calculate_amortization_payment(principal, monthly_rate, num_months):
    """Calculates the fixed monthly payment for an amortizing loan."""
    if monthly_rate == 0:
        # Avoid division by zero if interest rate is 0
        return principal / num_months if num_months > 0 else 0
    # Standard amortization formula
    payment = principal * (monthly_rate * (1 + monthly_rate)**num_months) / ((1 + monthly_rate)**num_months - 1)
    return payment

def calculate_outstanding_balance(principal, monthly_rate, num_months, payment_number):
    """Calculates the outstanding balance after a certain number of payments."""
    if payment_number >= num_months:
        return 0 # Loan fully paid off
    if monthly_rate == 0:
        # Linear principal reduction if no interest
        paid_principal = (principal / num_months) * payment_number
        return max(0, principal - paid_principal)

    # Calculate monthly payment using the amortization formula
    monthly_payment = calculate_amortization_payment(principal, monthly_rate, num_months)
    # Calculate outstanding balance using the formula B_n = P(1+r)^n - PMT[((1+r)^n - 1)/r]
    balance = principal * (1 + monthly_rate)**payment_number - \
              monthly_payment * (((1 + monthly_rate)**payment_number - 1) / monthly_rate)
    return max(0, balance) # Ensure balance doesn't go below zero

def calculate_exponential_survival_probs(annual_pd, max_months):
    """Calculates monthly survival probabilities assuming an exponential distribution."""
    if annual_pd >= 1.0:
        probs = [1.0] + [0.0] * max_months
        return probs
    if annual_pd <= 0:
        return [1.0] * (max_months + 1)
    hazard_lambda = -np.log(1 - annual_pd)
    months = np.arange(max_months + 1)
    survival_probs = np.exp(-hazard_lambda * months / 12.0)
    return survival_probs.tolist()

# --- Single-Period Functions (OBSOLETE - Kept for reference only) ---
# def calculate_baseline_profit_loss(): ...
# def simulate_strategy_1(recovery_rate_recall): ...
# def simulate_strategy_2(default_reduction_factor): ...

# --- Multi-Year Simulation Functions ---

def simulate_baseline_multiyear(simulation_years=SIMULATION_YEARS,
                                churn_rate=CHURN_RATE,
                                discount_rate=DISCOUNT_RATE):
    """Simulates the Baseline strategy over multiple years."""

    loans_per_bin = np.full(N_BINS, float(LOANS_PER_BIN))
    balance_per_loan = AVG_BALANCE

    yearly_net_pnl_after_funding = []
    total_absolute_loss_horizon = 0.0
    yearly_absolute_loss = []
    
    for year in range(simulation_years):
        current_total_loans = np.sum(loans_per_bin)
        if current_total_loans < 1: break

        # Apply Churn
        loans_per_bin *= (1 - churn_rate)
        current_total_loans = np.sum(loans_per_bin)
        if current_total_loans < 1: break
        start_of_year_portfolio_value = current_total_loans * balance_per_loan

        # Simulate Activity (Standard Balloon)
        year_gross_pnl = 0.0
        year_principal_loss = 0.0
        next_year_loans_per_bin = np.zeros(N_BINS)

        for i in range(N_BINS):
            n_loans = loans_per_bin[i]
            if n_loans < 1e-6: continue
            pd_rate = PD_RATES[i]
            expected_defaults = n_loans * pd_rate
            expected_non_defaults = n_loans * (1 - pd_rate)

            profit_non_default = expected_non_defaults * (balance_per_loan * INTEREST_RATE_MONTHLY * 11)
            loss_principal_default = expected_defaults * balance_per_loan * (1 - STANDARD_RECOVERY_RATE)

            year_gross_pnl += (profit_non_default - loss_principal_default)
            year_principal_loss += loss_principal_default
            next_year_loans_per_bin[i] = expected_non_defaults

        # Funding Cost
        year_funding_cost = start_of_year_portfolio_value * FUNDING_COST_RATE_ANNUAL * 1

        # Net P&L & Totals
        year_net_pnl_af = year_gross_pnl - year_funding_cost
        yearly_net_pnl_after_funding.append(year_net_pnl_af)
        yearly_absolute_loss.append(year_principal_loss)
        total_absolute_loss_horizon += year_principal_loss

        # Update State
        loans_per_bin = next_year_loans_per_bin

    # Final Metrics
    final_npv, yearly_cumulative_npv = calculate_npv(yearly_net_pnl_after_funding, discount_rate)

    print(f"Baseline Multi-Year ({simulation_years}yr): NPV=${final_npv:,.0f}, TotalAbsLoss=${total_absolute_loss_horizon:,.0f}")
    return final_npv, total_absolute_loss_horizon, yearly_cumulative_npv, yearly_net_pnl_after_funding, yearly_absolute_loss

def simulate_strategy1_multiyear(recovery_rate_recall,
                                simulation_years=SIMULATION_YEARS,
                                churn_rate=CHURN_RATE,
                                discount_rate=DISCOUNT_RATE):
    """Simulates Strategy 1 (Recall) over multiple years with churn and PD re-evaluation."""

    loans_per_bin = np.full(N_BINS, float(LOANS_PER_BIN))
    balance_per_loan = AVG_BALANCE

    yearly_net_pnl_after_funding = []
    total_absolute_loss_horizon = 0.0
    yearly_absolute_loss = []

    for year in range(simulation_years):
        current_total_loans = np.sum(loans_per_bin)
        if current_total_loans < 1: break

        # --- Start of Year: Apply Churn --- 
        loans_per_bin *= (1 - churn_rate)
        current_total_loans = np.sum(loans_per_bin)
        if current_total_loans < 1: break
        start_of_year_portfolio_value = current_total_loans * balance_per_loan

        # --- Simulate Activity for the Year (Strategy 1 Logic) --- 
        year_gross_pnl = 0.0
        year_principal_loss = 0.0
        next_year_loans_per_bin = np.zeros(N_BINS)

        for i in range(N_BINS):
            n_loans = loans_per_bin[i]
            if n_loans < 1e-6: continue

            pd_rate = PD_RATES[i]
            is_high_risk = i in HIGH_RISK_BINS

            expected_defaults = n_loans * pd_rate
            expected_non_defaults = n_loans * (1 - pd_rate)

            # Profit = Interest from survivors
            profit_non_default = expected_non_defaults * (balance_per_loan * INTEREST_RATE_MONTHLY * 11)

            # Loss = Principal loss from defaulters (using appropriate recovery rate)
            if is_high_risk:
                loss_principal_default = expected_defaults * balance_per_loan * (1 - recovery_rate_recall)
            else:
                loss_principal_default = expected_defaults * balance_per_loan * (1 - STANDARD_RECOVERY_RATE)

            year_gross_pnl += (profit_non_default - loss_principal_default)
            year_principal_loss += loss_principal_default

            # Survivors carry over to next year
            next_year_loans_per_bin[i] = expected_non_defaults

        # --- Calculate Funding Cost --- 
        year_funding_cost = start_of_year_portfolio_value * FUNDING_COST_RATE_ANNUAL * 1

        # --- Calculate Net P&L & Update Totals --- 
        year_net_pnl_gross = year_gross_pnl - year_principal_loss
        year_net_pnl_af = year_net_pnl_gross - year_funding_cost
        yearly_net_pnl_after_funding.append(year_net_pnl_af)
        yearly_absolute_loss.append(year_principal_loss)
        total_absolute_loss_horizon += year_principal_loss

        # --- Update State --- 
        loans_per_bin = next_year_loans_per_bin

    # --- Final Metrics --- 
    final_npv, yearly_cumulative_npv = calculate_npv(yearly_net_pnl_after_funding, discount_rate)

    # print(f" S1 Multi-Year (RR={recovery_rate_recall:.1f}, {simulation_years}yr): NPV=${final_npv:,.0f}, TotalAbsLoss=${total_absolute_loss_horizon:,.0f}")
    # Ensure all lists are returned
    return final_npv, total_absolute_loss_horizon, yearly_cumulative_npv, yearly_net_pnl_after_funding, yearly_absolute_loss

def _calculate_s2_low_risk_outcome(n_loans, start_balance, annual_pd):
    """Helper: Calculates total P&L/Loss over 24m amortization for LOW-RISK S2 loans.
    Applies rate increase, uses original PD.
    Returns total P&L and total Loss."""
    if n_loans <= 0: return 0.0, 0.0

    # Apply rate increase
    extended_annual_rate = INTEREST_RATE_ANNUAL + EXTENSION_RATE_INCREASE
    extended_monthly_rate = extended_annual_rate / 12
    principal = start_balance

    # Simulation based on reduced principal & 24m installment term
    monthly_payment = calculate_amortization_payment(principal, extended_monthly_rate, EXTENSION_MONTHS)
    total_interest_if_survives = (monthly_payment * EXTENSION_MONTHS) - principal

    # Use original PD for low-risk
    effective_annual_pd = max(0, min(annual_pd, 0.99999))
    
    survival_probs = calculate_exponential_survival_probs(effective_annual_pd, EXTENSION_MONTHS)
    
    expected_pnl_per_loan_ext = 0.0
    expected_loss_per_loan_ext = 0.0

    for t in range(1, EXTENSION_MONTHS + 1):
        prob_default_in_month_t = survival_probs[t-1] - survival_probs[t]
        if prob_default_in_month_t > 1e-9:
            outstanding_balance_at_t = calculate_outstanding_balance(principal, extended_monthly_rate, EXTENSION_MONTHS, t)
            payments_made = monthly_payment * (t - 1)
            interest_paid_before_t = payments_made - (principal - calculate_outstanding_balance(principal, extended_monthly_rate, EXTENSION_MONTHS, t-1)) if t > 1 else 0
            loss_on_principal_at_t = outstanding_balance_at_t * (1 - STANDARD_RECOVERY_RATE)
            pnl_if_default_at_t = interest_paid_before_t - loss_on_principal_at_t
            expected_pnl_per_loan_ext += prob_default_in_month_t * pnl_if_default_at_t
            expected_loss_per_loan_ext += prob_default_in_month_t * loss_on_principal_at_t

    prob_survival_total = survival_probs[EXTENSION_MONTHS]
    if prob_survival_total > 1e-9:
        pnl_if_survives = total_interest_if_survives
        expected_pnl_per_loan_ext += prob_survival_total * pnl_if_survives

    # P&L and Loss contribution from the extension period itself
    total_pnl_outcome = n_loans * expected_pnl_per_loan_ext
    total_loss_outcome = n_loans * expected_loss_per_loan_ext

    return total_pnl_outcome, total_loss_outcome

def _calculate_s2_high_risk_acceptor_outcome(n_accept, current_balance, pd_rate_annual, reduction_factor):
    """Helper: Calculates total P&L/Loss over 24m amortization for HIGH-RISK S2 ACCEPTORS.
    Applies haircut, rate increase, and PD reduction.
    Returns total P&L (including haircut impact) and total Loss (including haircut impact)."""
    if n_accept <= 0: return 0.0, 0.0

    # Apply haircut 
    initial_haircut_loss = n_accept * current_balance * PRINCIPAL_REDUCTION_FACTOR
    reduced_principal = current_balance * (1 - PRINCIPAL_REDUCTION_FACTOR)
    
    # Calculate increased rate for extension
    extended_annual_rate = INTEREST_RATE_ANNUAL + EXTENSION_RATE_INCREASE
    extended_monthly_rate = extended_annual_rate / 12

    # Simulation based on reduced principal & 24m installment term
    extended_monthly_payment = calculate_amortization_payment(reduced_principal, extended_monthly_rate, EXTENSION_MONTHS)
    total_interest_extension_if_survives = (extended_monthly_payment * EXTENSION_MONTHS) - reduced_principal

    # Apply PD reduction factor
    reduced_pd_annual = pd_rate_annual * (1 - reduction_factor)
    reduced_pd_annual = max(0, min(reduced_pd_annual, 0.99999))
    survival_probs = calculate_exponential_survival_probs(reduced_pd_annual, EXTENSION_MONTHS)
    
    expected_pnl_per_loan_ext = 0.0
    expected_loss_per_loan_ext = 0.0

    for t in range(1, EXTENSION_MONTHS + 1):
        prob_default_in_month_t = survival_probs[t-1] - survival_probs[t]
        if prob_default_in_month_t > 1e-9:
            outstanding_balance_at_t = calculate_outstanding_balance(reduced_principal, extended_monthly_rate, EXTENSION_MONTHS, t)
            payments_made = extended_monthly_payment * (t - 1)
            interest_paid_before_t = payments_made - (reduced_principal - calculate_outstanding_balance(reduced_principal, extended_monthly_rate, EXTENSION_MONTHS, t-1)) if t > 1 else 0
            loss_on_principal_at_t = outstanding_balance_at_t * (1 - STANDARD_RECOVERY_RATE)
            pnl_if_default_at_t = interest_paid_before_t - loss_on_principal_at_t
            expected_pnl_per_loan_ext += prob_default_in_month_t * pnl_if_default_at_t
            expected_loss_per_loan_ext += prob_default_in_month_t * loss_on_principal_at_t

    prob_survival_total = survival_probs[EXTENSION_MONTHS]
    if prob_survival_total > 1e-9:
        pnl_if_survives = total_interest_extension_if_survives
        expected_pnl_per_loan_ext += prob_survival_total * pnl_if_survives

    # P&L and Loss contribution from the extension period itself
    pnl_extension_period = n_accept * expected_pnl_per_loan_ext
    loss_extension_period = n_accept * expected_loss_per_loan_ext

    # Combine with haircut impact
    total_pnl_outcome = pnl_extension_period - initial_haircut_loss # Start with haircut loss
    total_loss_outcome = loss_extension_period + initial_haircut_loss # Add haircut loss

    return total_pnl_outcome, total_loss_outcome

def simulate_strategy2_multiyear(default_reduction_factor,
                                simulation_years=SIMULATION_YEARS,
                                churn_rate=CHURN_RATE,
                                discount_rate=DISCOUNT_RATE):
    """Simulates Strategy 2 (Mandatory 24m Amortization for All, w/ Haircut/Refusal/PD-Reduct for High-Risk) over 2 years."""

    # Strategy 2 simulation runs only for 2 years
    actual_simulation_years = 2 

    yearly_net_pnl_after_funding = []
    total_absolute_loss_horizon = 0.0
    yearly_absolute_loss = []
    value_entering_yr1_ext = 0.0 # Value entering extension (for Yr1 funding)
    value_entering_yr2_ext = 0.0 # Value entering Yr2 of extension (for Yr2 funding)

    # --- Year 1 --- 
    year = 0
    current_year_num = year + 1

    # Initial state
    loans_per_bin_start = np.full(N_BINS, float(LOANS_PER_BIN))

    # --- Apply Churn (at start of Year 1 only) --- 
    loans_per_bin_after_churn = loans_per_bin_start * (1 - churn_rate)

    # --- Calculate Total Expected P&L/Loss over 24m for the whole starting cohort --- 
    total_pnl_allocated_yr1 = 0.0
    total_loss_allocated_yr1 = 0.0
    total_pnl_allocated_yr2 = 0.0
    total_loss_allocated_yr2 = 0.0

    for i in range(N_BINS):
        n_loans = loans_per_bin_after_churn[i]
        start_balance = AVG_BALANCE # Original balance
        annual_pd = PD_RATES[i]
        is_high_risk = i in HIGH_RISK_BINS
        if n_loans > 1e-6:
            value_entering_yr1_ext += n_loans * start_balance # Track value before any S2 action

            if not is_high_risk:
                # Low Risk: Calculate 24m outcome & allocate
                pnl_24m_total, loss_24m_total = _calculate_s2_low_risk_outcome(n_loans, start_balance, annual_pd)
                total_pnl_allocated_yr1 += pnl_24m_total / 2.0
                total_loss_allocated_yr1 += loss_24m_total / 2.0
                total_pnl_allocated_yr2 += pnl_24m_total / 2.0
                total_loss_allocated_yr2 += loss_24m_total / 2.0
                value_entering_yr2_ext += n_loans * start_balance # Track these loans for Yr2 funding
            else:
                # High Risk: Apply Refusal/Acceptance
                # Refusers (resolve Year 1)
                n_refuse = n_loans * REFUSAL_PROBABILITY
                if n_refuse > 0:
                    expected_defaults_refuse = n_refuse * annual_pd
                    expected_non_defaults_refuse = n_refuse * (1 - annual_pd)
                    profit_refuse = expected_non_defaults_refuse * (start_balance * INTEREST_RATE_MONTHLY * 11)
                    loss_refuse = expected_defaults_refuse * start_balance * (1 - STANDARD_RECOVERY_RATE)
                    total_pnl_allocated_yr1 += (profit_refuse - loss_refuse)
                    total_loss_allocated_yr1 += loss_refuse
                
                # Acceptors (allocate over Year 1 & 2)
                n_accept = n_loans * (1 - REFUSAL_PROBABILITY)
                if n_accept > 0:
                    # Loss from haircut occurs Yr 1
                    initial_haircut_loss = n_accept * start_balance * PRINCIPAL_REDUCTION_FACTOR
                    total_loss_allocated_yr1 += initial_haircut_loss
                    
                    # Calculate the P&L/Loss *excluding* haircut (helper includes it)
                    pnl_24m_outcome, loss_24m_outcome = _calculate_s2_high_risk_acceptor_outcome(n_accept, start_balance, annual_pd, default_reduction_factor)
                    
                    # Allocate the total outcomes over Yr1 and Yr2
                    total_pnl_allocated_yr1 += pnl_24m_outcome / 2.0
                    total_loss_allocated_yr1 += (loss_24m_outcome - initial_haircut_loss) / 2.0 # Remove haircut from allocation
                    total_pnl_allocated_yr2 += pnl_24m_outcome / 2.0
                    total_loss_allocated_yr2 += (loss_24m_outcome - initial_haircut_loss) / 2.0
                    value_entering_yr2_ext += n_accept * start_balance * (1-PRINCIPAL_REDUCTION_FACTOR) # Track reduced value for Yr2 funding

    # --- Year 1 Calculations --- 
    pnl_year1 = total_pnl_allocated_yr1
    loss_year1 = total_loss_allocated_yr1

    # Funding Cost for Year 1 (based on value entering the year)
    funding_cost_year1 = value_entering_yr1_ext * FUNDING_COST_RATE_ANNUAL * 1

    # --- Calculate Net P&L & Update Horizon Loss for Year 1 --- 
    net_pnl_af_year1 = pnl_year1 - funding_cost_year1
    yearly_net_pnl_after_funding.append(net_pnl_af_year1)
    total_absolute_loss_horizon += loss_year1
    yearly_absolute_loss.append(loss_year1)

    # --- Year 2 --- 
    year = 1
    current_year_num = year + 1
    if current_year_num > actual_simulation_years: # Should match actual_simulation_years
        pass # No Year 3 etc. for this strategy
    else: 
        # --- Allocate P&L/Loss to Year 2 --- 
        pnl_year2 = total_pnl_allocated_yr2
        loss_year2 = total_loss_allocated_yr2

        # --- Calculate Funding Cost for Year 2 --- 
        # Base cost on the value of loans still active in Year 2
        funding_cost_year2 = value_entering_yr2_ext * FUNDING_COST_RATE_ANNUAL * 1 
        
        # --- Calculate Net P&L & Update Horizon Loss for Year 2 --- 
        net_pnl_af_year2 = pnl_year2 - funding_cost_year2
        yearly_net_pnl_after_funding.append(net_pnl_af_year2)
        total_absolute_loss_horizon += loss_year2
        yearly_absolute_loss.append(loss_year2)

    # --- Final Metrics --- 
    final_npv, yearly_cumulative_npv = calculate_npv(yearly_net_pnl_after_funding, discount_rate)

    # print(f" S2 Multi-Year (RF={default_reduction_factor:.1f}, {simulation_years}yr): NPV=${final_npv:,.0f}, TotalAbsLoss=${total_absolute_loss_horizon:,.0f}")
    # Return final NPV, total loss, and potentially yearly cumulative NPV for plotting
    return final_npv, total_absolute_loss_horizon, yearly_cumulative_npv, yearly_net_pnl_after_funding, yearly_absolute_loss

def simulate_strategy_2(default_reduction_factor):
    """Simulates single-period Strategy 2 (24m Installment Ext), returns P&L AF and Abs Loss."""
    # NOTE: This function is likely now obsolete, replaced by simulate_strategy2_multiyear
    # Keeping it here just in case, but it shouldn't be called in the main flow.
    total_gross_pnl = 0
    total_expected_loss_on_principal = 0
    total_funding_cost = 0
    n_high_risk_loans = len(HIGH_RISK_BINS) * LOANS_PER_BIN
    n_low_risk_loans = N_LOANS - n_high_risk_loans
    value_high_risk = n_high_risk_loans * AVG_BALANCE
    value_low_risk = n_low_risk_loans * AVG_BALANCE
    # --- Low Risk Bins (Baseline Logic) ---
    pnl_low_risk = 0
    loss_low_risk = 0
    for i in range(N_BINS):
        if i not in HIGH_RISK_BINS:
            n_loans_in_bin = LOANS_PER_BIN
            pd_rate_annual = PD_RATES[i]
            expected_defaults = n_loans_in_bin * pd_rate_annual
            expected_non_defaults = n_loans_in_bin * (1 - pd_rate_annual)
            profit_non_default = expected_non_defaults * TOTAL_INTEREST_ORIGINAL
            loss_principal_default = expected_defaults * AVG_BALANCE * (1 - STANDARD_RECOVERY_RATE)
            pnl_low_risk += (profit_non_default - loss_principal_default)
            loss_low_risk += loss_principal_default
    total_gross_pnl += pnl_low_risk
    total_expected_loss_on_principal += loss_low_risk
    total_funding_cost += value_low_risk * FUNDING_COST_RATE_ANNUAL * 1
    # --- High Risk Bins ---
    pnl_high_risk = 0
    loss_high_risk = 0
    funding_duration_high_risk = 0
    n_loans_processed_high_risk = 0
    for i in HIGH_RISK_BINS:
        n_loans_in_bin = LOANS_PER_BIN
        pd_rate_annual = PD_RATES[i]
        # Refusal Group
        n_refuse = n_loans_in_bin * REFUSAL_PROBABILITY
        expected_defaults_refuse = n_refuse * pd_rate_annual
        expected_non_defaults_refuse = n_refuse * (1 - pd_rate_annual)
        profit_non_default_refuse = expected_non_defaults_refuse * TOTAL_INTEREST_ORIGINAL
        loss_principal_default_refuse = expected_defaults_refuse * AVG_BALANCE * (1 - STANDARD_RECOVERY_RATE)
        pnl_high_risk += (profit_non_default_refuse - loss_principal_default_refuse)
        loss_high_risk += loss_principal_default_refuse
        funding_duration_high_risk += n_refuse * 1.0
        n_loans_processed_high_risk += n_refuse
        # Acceptance Group
        n_accept = n_loans_in_bin * (1 - REFUSAL_PROBABILITY)
        if n_accept > 0:
            initial_haircut_loss = n_accept * AVG_BALANCE * PRINCIPAL_REDUCTION_FACTOR
            reduced_principal = AVG_BALANCE * (1 - PRINCIPAL_REDUCTION_FACTOR)
            loss_high_risk += initial_haircut_loss
            pnl_high_risk -= initial_haircut_loss

            # Calculate increased rate for extension
            extended_annual_rate = INTEREST_RATE_ANNUAL + EXTENSION_RATE_INCREASE
            extended_monthly_rate = extended_annual_rate / 12

            # Simulation based on reduced principal & 24m installment term
            extended_monthly_payment = calculate_amortization_payment(reduced_principal, extended_monthly_rate, EXTENSION_MONTHS)
            total_interest_extension_if_survives = (extended_monthly_payment * EXTENSION_MONTHS) - reduced_principal

            reduced_pd_annual = pd_rate_annual * (1 - default_reduction_factor)
            reduced_pd_annual = max(0, min(reduced_pd_annual, 0.99999))
            survival_probs = calculate_exponential_survival_probs(reduced_pd_annual, EXTENSION_MONTHS)
            expected_pnl_per_loan_accept = 0
            expected_loss_per_loan_accept = 0
            bin_expected_duration_accept = 0
            for t in range(1, EXTENSION_MONTHS + 1):
                prob_default_in_month_t = survival_probs[t-1] - survival_probs[t]
                if prob_default_in_month_t > 1e-9:
                    outstanding_balance_at_t = calculate_outstanding_balance(reduced_principal, extended_monthly_rate, EXTENSION_MONTHS, t)
                    payments_made = extended_monthly_payment * (t - 1)
                    interest_paid_before_t = payments_made - (reduced_principal - calculate_outstanding_balance(reduced_principal, extended_monthly_rate, EXTENSION_MONTHS, t-1)) if t > 1 else 0
                    loss_on_principal_at_t = outstanding_balance_at_t * (1 - STANDARD_RECOVERY_RATE)
                    pnl_if_default_at_t = interest_paid_before_t - loss_on_principal_at_t
                    expected_pnl_per_loan_accept += prob_default_in_month_t * pnl_if_default_at_t
                    expected_loss_per_loan_accept += prob_default_in_month_t * loss_on_principal_at_t
                    bin_expected_duration_accept += prob_default_in_month_t * (t / 12.0)
            prob_survival_total = survival_probs[EXTENSION_MONTHS]
            if prob_survival_total > 1e-9:
                pnl_if_survives = total_interest_extension_if_survives
                expected_pnl_per_loan_accept += prob_survival_total * pnl_if_survives
                bin_expected_duration_accept += prob_survival_total * (EXTENSION_MONTHS / 12.0)
            pnl_high_risk += n_accept * expected_pnl_per_loan_accept
            loss_high_risk += n_accept * expected_loss_per_loan_accept
            funding_duration_high_risk += n_accept * bin_expected_duration_accept
            n_loans_processed_high_risk += n_accept
    total_gross_pnl += pnl_high_risk
    total_expected_loss_on_principal += loss_high_risk
    avg_funding_duration_high_risk = funding_duration_high_risk / n_loans_processed_high_risk if n_loans_processed_high_risk > 0 else 0
    total_funding_cost += value_high_risk * FUNDING_COST_RATE_ANNUAL * avg_funding_duration_high_risk
    net_profit_loss_after_funding = total_gross_pnl - total_funding_cost
    return net_profit_loss_after_funding, total_expected_loss_on_principal

# --- Multi-Year Simulation for Strategy 3 ---
def simulate_strategy3_multiyear(default_reduction_factor,
                                simulation_years=SIMULATION_YEARS,
                                churn_rate=CHURN_RATE,
                                discount_rate=DISCOUNT_RATE):
    """Simulates Strategy 3 over multiple years with churn, haircut, refusal, and indefinite extension."""

    # Initial state: Track loans per bin and their *current* average balance
    # Using a dictionary per bin for {count, balance}
    portfolio_state = [{'count': float(LOANS_PER_BIN), 'balance': AVG_BALANCE} for _ in range(N_BINS)]

    yearly_net_pnl_after_funding = []
    total_absolute_loss_horizon = 0.0
    yearly_absolute_loss = []
    yearly_end_of_year_balance = [] # Track balance over time

    for year in range(simulation_years):
        current_total_loans = sum(bin_state['count'] for bin_state in portfolio_state)
        if current_total_loans < 1: break # Stop if portfolio is empty

        # --- Start of Year State --- Apply Churn ---
        # Churn happens at the START, affecting loans available for the year.
        # Churn removes loans proportionally from each bin count. Balance assumed AVG.
        for i in range(N_BINS):
            portfolio_state[i]['count'] *= (1 - churn_rate)

        current_total_loans = sum(bin_state['count'] for bin_state in portfolio_state)
        if current_total_loans < 1: break

        start_of_year_portfolio_value = sum(bin_state['count'] * bin_state['balance'] for bin_state in portfolio_state)

        # --- Simulate Activity and Calculate P&L/Loss for the Year ---
        year_gross_pnl = 0
        year_principal_loss = 0
        next_year_portfolio_state = [{'count': 0.0, 'balance': 0.0} for _ in range(N_BINS)] # Track state for next year

        for i in range(N_BINS):
            n_loans = portfolio_state[i]['count']
            current_balance = portfolio_state[i]['balance']
            if n_loans < 1e-6: continue

            pd_rate_annual = PD_RATES[i] # Reapply original PD rate for bin classification

            if i not in HIGH_RISK_BINS:
                # --- Low Risk Bins: Baseline Balloon Logic for this year ---
                expected_defaults = n_loans * pd_rate_annual
                expected_non_defaults = n_loans * (1 - pd_rate_annual)

                # P&L = Interest from survivors - Loss from defaults
                profit_non_default = expected_non_defaults * (current_balance * INTEREST_RATE_MONTHLY * 11)
                loss_principal_default = expected_defaults * current_balance * (1 - STANDARD_RECOVERY_RATE)

                year_gross_pnl += (profit_non_default - loss_principal_default)
                year_principal_loss += loss_principal_default

                # Survivors continue to next year in the same bin with same balance
                next_year_portfolio_state[i]['count'] = expected_non_defaults
                next_year_portfolio_state[i]['balance'] = current_balance

            else:
                # --- High Risk Bins: Apply Strategy 3 Logic for this year ---
                bin_pnl_high_risk = 0
                bin_loss_high_risk = 0

                # --- Refusal Group ---
                n_refuse = n_loans * REFUSAL_PROBABILITY
                if n_refuse > 0:
                    expected_defaults_refuse = n_refuse * pd_rate_annual # Default based on *original* PD
                    expected_non_defaults_refuse = n_refuse * (1 - pd_rate_annual) # Payoff original

                    profit_refuse = expected_non_defaults_refuse * (current_balance * INTEREST_RATE_MONTHLY * 11)
                    loss_refuse = expected_defaults_refuse * current_balance * (1 - STANDARD_RECOVERY_RATE)

                    bin_pnl_high_risk += (profit_refuse - loss_refuse)
                    bin_loss_high_risk += loss_refuse
                    # Refusers are resolved this year, don't carry over under S3 extension.

                # --- Acceptance Group ---
                n_accept = n_loans * (1 - REFUSAL_PROBABILITY)
                if n_accept > 0:
                    # Apply haircut immediately
                    initial_haircut_loss = n_accept * current_balance * PRINCIPAL_REDUCTION_FACTOR
                    reduced_principal = current_balance * (1 - PRINCIPAL_REDUCTION_FACTOR)

                    # Haircut is direct loss and P&L reduction
                    bin_loss_high_risk += initial_haircut_loss
                    bin_pnl_high_risk -= initial_haircut_loss

                    # Apply 12m Balloon Extension logic (1/3 mid, 2/3 end default split)
                    reduced_pd_annual_ext = pd_rate_annual * (1 - default_reduction_factor)
                    reduced_pd_annual_ext = max(0, min(reduced_pd_annual_ext, 0.99999))

                    prob_default_total_ext = reduced_pd_annual_ext
                    prob_default_midterm_ext = (1/3) * prob_default_total_ext
                    prob_default_endterm_ext = (2/3) * prob_default_total_ext
                    prob_survival_ext = 1.0 - prob_default_total_ext

                    expected_defaults_midterm = n_accept * prob_default_midterm_ext
                    expected_defaults_endterm = n_accept * prob_default_endterm_ext
                    expected_non_defaults = n_accept * prob_survival_ext # These survive the extension

                    # Calculate interest using the increased rate for the extension
                    extended_annual_rate = INTEREST_RATE_ANNUAL + EXTENSION_RATE_INCREASE
                    extended_monthly_rate = extended_annual_rate / 12
                    interest_per_loan_ext = reduced_principal * extended_monthly_rate * 11

                    loss_per_default_ext = reduced_principal * (1 - STANDARD_RECOVERY_RATE)

                    # P&L from extension period
                    pnl_midterm = - expected_defaults_midterm * loss_per_default_ext
                    pnl_endterm = expected_defaults_endterm * (interest_per_loan_ext - loss_per_default_ext)
                    pnl_survival = expected_non_defaults * interest_per_loan_ext
                    pnl_extension_period = pnl_midterm + pnl_endterm + pnl_survival
                    bin_pnl_high_risk += pnl_extension_period

                    # Loss from extension period
                    loss_midterm = expected_defaults_midterm * loss_per_default_ext
                    loss_endterm = expected_defaults_endterm * loss_per_default_ext
                    loss_extension_period = loss_midterm + loss_endterm
                    bin_loss_high_risk += loss_extension_period

                    # Survivors carry over to next year in the same bin, but with reduced balance
                    next_year_portfolio_state[i]['count'] = expected_non_defaults
                    next_year_portfolio_state[i]['balance'] = reduced_principal

                # Accumulate P&L and Loss for the bin
                year_gross_pnl += bin_pnl_high_risk
                year_principal_loss += bin_loss_high_risk

        # --- Calculate Funding Cost for the Year ---
        # Simple approach: Cost based on start-of-year value, 1-year duration for all active loans
        year_funding_cost = start_of_year_portfolio_value * FUNDING_COST_RATE_ANNUAL * 1

        # --- Calculate Net P&L for the Year ---
        year_net_pnl_af = year_gross_pnl - year_funding_cost
        yearly_net_pnl_after_funding.append(year_net_pnl_af)
        yearly_absolute_loss.append(year_principal_loss)
        total_absolute_loss_horizon += year_principal_loss

        # Calculate and store end-of-year balance (before next year's churn)
        end_of_year_balance = sum(bin_state['count'] * bin_state['balance'] for bin_state in next_year_portfolio_state)
        yearly_end_of_year_balance.append(end_of_year_balance)

        # --- Update Portfolio State for Next Year ---
        portfolio_state = next_year_portfolio_state

    # --- Calculate Final Metrics ---
    final_npv, yearly_cumulative_npv = calculate_npv(yearly_net_pnl_after_funding, discount_rate)

    # print(f" S3 Multi-Year (RF={default_reduction_factor:.1f}, {simulation_years}yr): NPV=${final_npv:,.0f}, TotalAbsLoss=${total_absolute_loss_horizon:,.0f}")
    # Return final metrics and yearly details for potential export
    return (final_npv, total_absolute_loss_horizon, yearly_end_of_year_balance, 
            yearly_cumulative_npv, yearly_net_pnl_after_funding, yearly_absolute_loss)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Running Multi-Year Simulations for All Strategies ---")
    (baseline_npv, baseline_loss, baseline_cum_npv, 
     baseline_yearly_pnl, baseline_yearly_loss) = simulate_baseline_multiyear()

    strategy1_results_npv = {}
    strategy1_results_loss = {}
    strategy1_results_cum_npv = {}
    strategy1_yearly_pnl = {}
    strategy1_yearly_loss = {}
    print(f"\nSimulating Strategy 1 (Recall - Multi-Year, {SIMULATION_YEARS} years, {CHURN_RATE*100}% churn)...")
    for recovery_rate in RECOVERY_RATES_STRAT1:
        npv, loss, yearly_cum_npv, yearly_pnl, yearly_loss = simulate_strategy1_multiyear(recovery_rate)
        strategy1_results_npv[recovery_rate] = npv
        strategy1_results_loss[recovery_rate] = loss
        strategy1_results_cum_npv[recovery_rate] = yearly_cum_npv
        strategy1_yearly_pnl[recovery_rate] = yearly_pnl
        strategy1_yearly_loss[recovery_rate] = yearly_loss
        print(f"  S1 Multi-Year (RR={recovery_rate:.1f}): NPV=${npv:,.0f}, TotalAbsLoss=${loss:,.0f}")

    strategy2_results_npv = {}
    strategy2_results_loss = {}
    strategy2_results_cumulative_npv = {}
    strategy2_yearly_pnl = {}
    strategy2_yearly_loss = {}
    print(f"\nSimulating Strategy 2 (24m Installment Ext - Multi-Year, {SIMULATION_YEARS} years, {CHURN_RATE*100}% churn)...")
    for reduction_factor in DEFAULT_REDUCTION_FACTORS:
        npv, loss, yearly_cum_npv, yearly_pnl, yearly_loss_s2 = simulate_strategy2_multiyear(reduction_factor)
        strategy2_results_npv[reduction_factor] = npv
        strategy2_results_loss[reduction_factor] = loss
        strategy2_results_cumulative_npv[reduction_factor] = yearly_cum_npv
        strategy2_yearly_pnl[reduction_factor] = yearly_pnl
        strategy2_yearly_loss[reduction_factor] = yearly_loss_s2
        print(f"  S2 Multi-Year (RF={reduction_factor:.1f}): NPV=${npv:,.0f}, TotalAbsLoss=${loss:,.0f}")

    print(f"\n--- Running Multi-Year Simulation for Strategy 3 ({SIMULATION_YEARS} years, {CHURN_RATE*100}% churn) ---")
    strategy3_results_npv = {}
    strategy3_results_loss = {}
    strategy3_results_balances = {}
    strategy3_results_cumulative_npv = {}
    strategy3_yearly_pnl_details = {}
    strategy3_yearly_loss_details = {}
    for reduction_factor in DEFAULT_REDUCTION_FACTORS:
        (final_npv, total_loss, yearly_balances, 
         yearly_cum_npv, yearly_pnl, yearly_loss) = simulate_strategy3_multiyear(reduction_factor)

        strategy3_results_npv[reduction_factor] = final_npv
        strategy3_results_loss[reduction_factor] = total_loss
        strategy3_results_balances[reduction_factor] = yearly_balances
        strategy3_results_cumulative_npv[reduction_factor] = yearly_cum_npv
        strategy3_yearly_pnl_details[reduction_factor] = yearly_pnl
        strategy3_yearly_loss_details[reduction_factor] = yearly_loss
        print(f"  S3 Multi-Year (RF={reduction_factor:.1f}): Final NPV=${final_npv:,.0f}, TotalAbsLoss=${total_loss:,.0f}")

    print("\n--- Simulation Analysis Complete ---")

    # --- Export Results to CSV --- 
    print("\nExporting results to CSV...")

    # Create base path for results
    results_dir = "simulation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    def save_yearly_results(data_dict, filename_base, results_dir="simulation_results"):
        # Handle cases where inner lists might have different lengths if simulations stop early
        df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
        # Pad shorter columns with NaN if necessary (though likely not needed with fixed horizons now)
        # df = df.reindex(range(max(len(lst) for lst in data_dict.values() if lst is not None)))
        df.index.name = 'Year' # Set index starting from 0
        df.index = df.index + 1 # Make index 1-based for readability
        path = os.path.join(results_dir, f"{filename_base}.csv")
        df.to_csv(path, float_format='%.2f')
        print(f"  Saved yearly data to {path}")

    # Baseline Yearly
    df_baseline_yearly = pd.DataFrame({'PnL_AF': baseline_yearly_pnl, 'Abs_Loss': baseline_yearly_loss})
    df_baseline_yearly.index.name = 'Year'
    df_baseline_yearly.index = df_baseline_yearly.index + 1
    baseline_path = os.path.join(results_dir, "results_baseline_yearly.csv")
    df_baseline_yearly.to_csv(baseline_path, float_format='%.2f')
    print(f"  Saved Baseline yearly data to {baseline_path}")

    # Strategy 1 Yearly Results
    save_yearly_results(strategy1_yearly_pnl, "results_s1_yearly_pnl_af")
    save_yearly_results(strategy1_yearly_loss, "results_s1_yearly_loss")

    # Strategy 2 Yearly Results
    save_yearly_results(strategy2_yearly_pnl, "results_s2_yearly_pnl_af")
    save_yearly_results(strategy2_yearly_loss, "results_s2_yearly_loss")

    # Strategy 3 Yearly P&L
    save_yearly_results(strategy3_yearly_pnl_details, "results_s3_yearly_pnl_af")

    # Strategy 3 Yearly Loss
    save_yearly_results(strategy3_yearly_loss_details, "results_s3_yearly_loss")

    # Optional: Save final summary metrics (NPV, Total Loss)
    summary_metrics = []
    summary_metrics.append({'Strategy': 'Baseline', 'Parameter': 'N/A', 'Value': np.nan, 'NPV': baseline_npv, 'Total_Loss': baseline_loss})
    for param, val in strategy1_results_npv.items(): summary_metrics.append({'Strategy': 'S1', 'Parameter': 'Recovery_Rate', 'Value': param, 'NPV': val, 'Total_Loss': strategy1_results_loss[param]})
    for param, val in strategy2_results_npv.items(): summary_metrics.append({'Strategy': 'S2', 'Parameter': 'Reduction_Factor', 'Value': param, 'NPV': val, 'Total_Loss': strategy2_results_loss[param]})
    for param, val in strategy3_results_npv.items(): summary_metrics.append({'Strategy': 'S3', 'Parameter': 'Reduction_Factor', 'Value': param, 'NPV': val, 'Total_Loss': strategy3_results_loss[param]})
    df_summary = pd.DataFrame(summary_metrics)
    summary_path = os.path.join(results_dir, "results_final_summary.csv")
    df_summary.to_csv(summary_path, index=False, float_format='%.2f')
    print(f"  Saved final summary metrics to {summary_path}")

    # --- Visualization (Comparing Multi-Year S1, S2, S3 vs Multi-Year Baseline) --- 
    plt.style.use('seaborn-v0_8-darkgrid')

    # --- P&L / NPV Plot ---
    fig_pnl, (ax_pnl1, ax_pnl2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig_pnl.suptitle(f'Strategy Comparison ({SIMULATION_YEARS}yr NPV/Loss Horizon)', fontsize=16)

    # Baseline (Multi-Year NPV as horizontal line)
    ax_pnl1.axhline(baseline_npv, color='grey', linestyle='--', linewidth=2, label=f'Baseline NPV (${baseline_npv:,.0f})')
    ax_pnl2.axhline(baseline_npv, color='grey', linestyle='--', linewidth=2, label=f'Baseline NPV')

    # Strategy 1 (Multi-Year NPV vs Recovery Rate)
    s1_rates_keys = list(strategy1_results_npv.keys())
    s1_npv_values = list(strategy1_results_npv.values())
    ax_pnl1.plot(s1_rates_keys, s1_npv_values, marker='o', linestyle='-', linewidth=2, markersize=8, color='skyblue', label='Strategy 1 (Recall) NPV')
    ax_pnl1.set_xlabel('Recovery Rate (Strategy 1)', fontsize=12)
    ax_pnl1.set_ylabel(f'{SIMULATION_YEARS}-Year NPV ($)', fontsize=12)
    ax_pnl1.set_title(f'Strategy 1 NPV vs. Baseline', fontsize=14)
    ax_pnl1.legend()
    ax_pnl1.grid(True)
    ax_pnl1.ticklabel_format(style='plain', axis='y')

    # Strategy 2 (Multi-Year NPV vs Reduction Factor)
    s2_factors_keys = list(strategy2_results_npv.keys())
    s2_npv_values = list(strategy2_results_npv.values())
    s3_npv_values = list(strategy3_results_npv.values())
    ax_pnl2.plot(s2_factors_keys, s2_npv_values, marker='s', linestyle='-', linewidth=2, markersize=8, color='lightcoral', label=f'Strategy 2 ({SIMULATION_YEARS}yr NPV)')
    ax_pnl2.plot(s2_factors_keys, s3_npv_values, marker='^', linestyle=':', linewidth=2, markersize=8, color='mediumpurple', label=f'Strategy 3 ({SIMULATION_YEARS}yr NPV)')
    ax_pnl2.set_xlabel('Default Reduction Factor (Strategy 2 & 3)', fontsize=12)
    ax_pnl2.set_ylabel(f'{SIMULATION_YEARS}-Year NPV ($)', fontsize=12)
    ax_pnl2.set_title(f'Strategies 2 & 3 NPV vs. Baseline', fontsize=14)
    ax_pnl2.legend()
    ax_pnl2.grid(True)
    ax_pnl2.ticklabel_format(style='plain', axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show() # Show plots separately or together

    # --- Absolute Loss Plot ---
    fig_loss, (ax_loss1, ax_loss2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig_loss.suptitle(f'Strategy Comparison ({SIMULATION_YEARS}yr Total Absolute Loss Horizon)', fontsize=16)

    # Baseline Loss (Multi-Year Total Loss as horizontal line)
    ax_loss1.axhline(baseline_loss, color='grey', linestyle='--', linewidth=2, label=f'Baseline Total Loss (${baseline_loss:,.0f})')
    ax_loss2.axhline(baseline_loss, color='grey', linestyle='--', linewidth=2, label=f'Baseline Total Loss')

    # Strategy 1 Loss (Multi-Year Total Loss vs Recovery Rate)
    s1_loss_values = list(strategy1_results_loss.values())
    ax_loss1.plot(s1_rates_keys, s1_loss_values, marker='o', linestyle='-', linewidth=2, markersize=8, color='orange', label='Strategy 1 (Recall) Loss')
    ax_loss1.set_xlabel('Recovery Rate (Strategy 1)', fontsize=12)
    ax_loss1.set_ylabel(f'{SIMULATION_YEARS}-Year Total Absolute Loss ($)', fontsize=12)
    ax_loss1.set_title(f'Strategy 1 Total Loss vs. Baseline', fontsize=14)
    ax_loss1.legend()
    ax_loss1.grid(True)
    ax_loss1.ticklabel_format(style='plain', axis='y')

    # Strategy 2 Loss (Multi-Year Total Loss vs Reduction Factor)
    s2_loss_values = list(strategy2_results_loss.values())
    s3_total_loss_values = list(strategy3_results_loss.values())
    ax_loss2.plot(s2_factors_keys, s2_loss_values, marker='s', linestyle='-', linewidth=2, markersize=8, color='red', label=f'Strategy 2 ({SIMULATION_YEARS}yr Total Loss)')
    ax_loss2.plot(s2_factors_keys, s3_total_loss_values, marker='^', linestyle=':', linewidth=2, markersize=8, color='darkorange', label=f'Strategy 3 ({SIMULATION_YEARS}yr Total Loss)')
    ax_loss2.set_xlabel('Default Reduction Factor (Strategy 2 & 3)', fontsize=12)
    ax_loss2.set_ylabel(f'{SIMULATION_YEARS}-Year Total Absolute Loss ($)', fontsize=12)
    ax_loss2.set_title(f'Strategies 2 & 3 Total Loss vs. Baseline', fontsize=14)
    ax_loss2.legend()
    ax_loss2.grid(True)
    ax_loss2.ticklabel_format(style='plain', axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Portfolio Balance Evolution Plot (Strategy 3 Multi-Year) ---
    fig_balance, ax_balance = plt.subplots(figsize=(12, 7))
    fig_balance.suptitle(f'Strategy 3: Portfolio Balance Evolution over {SIMULATION_YEARS} Years ({CHURN_RATE*100}% Churn)', fontsize=14)

    # Use a colormap for different reduction factors
    colors = plt.cm.viridis(np.linspace(0, 1, len(DEFAULT_REDUCTION_FACTORS)))

    for idx, reduction_factor in enumerate(DEFAULT_REDUCTION_FACTORS):
        balances = strategy3_results_balances[reduction_factor]
        years = range(len(balances)) # Use actual length in case simulation stops early
        ax_balance.plot(years, balances, marker='.', linestyle='-', linewidth=2, markersize=8, color=colors[idx], label=f'RF = {reduction_factor:.1f}')

    ax_balance.set_xlabel('Year', fontsize=12)
    ax_balance.set_ylabel('End-of-Year Portfolio Balance ($)', fontsize=12)
    ax_balance.set_title('Projected Balance under Different Default Reduction Factors', fontsize=12)
    ax_balance.legend(title="Reduction Factor")
    ax_balance.grid(True)
    ax_balance.ticklabel_format(style='plain', axis='y')
    ax_balance.set_xticks(range(0, SIMULATION_YEARS, max(1, SIMULATION_YEARS // 10))) # Adjust x-ticks for clarity

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # Show all plots now

    # --- Cumulative NPV Plot ---
    fig_cum_npv, ax_cum_npv = plt.subplots(figsize=(12, 7))
    fig_cum_npv.suptitle(f'Cumulative NPV Comparison over {SIMULATION_YEARS} Years', fontsize=14)

    # Plot Baseline Cumulative NPV Timeseries
    plot_years_base = range(1, len(baseline_cum_npv) + 1)
    ax_cum_npv.plot(plot_years_base, baseline_cum_npv, marker='D', linestyle='--', linewidth=2, markersize=6, color='grey', label='Baseline Cum. NPV', zorder=5)

    # Plot Strategy 1 Final NPV Range (as bar at end year)
    s1_min_npv = min(strategy1_results_npv.values())
    s1_max_npv = max(strategy1_results_npv.values())
    ax_cum_npv.plot([SIMULATION_YEARS, SIMULATION_YEARS], [s1_min_npv, s1_max_npv], color='skyblue', 
                    linewidth=6, alpha=0.7, label=f'S1 {SIMULATION_YEARS}yr NPV Range')

    # Plot Strategy 2 Range (constant horizontal band)
    s2_min_npv = min(strategy2_results_npv.values())
    s2_max_npv = max(strategy2_results_npv.values())
    ax_cum_npv.plot([2, 2], [s2_min_npv, s2_max_npv], color='lightcoral', linewidth=6, alpha=0.7, label=f'Strategy 2 Range (Total P&L AF)')

    # Plot Strategy 3 Cumulative NPV Timeseries
    colors = plt.cm.viridis(np.linspace(0, 1, len(DEFAULT_REDUCTION_FACTORS)))
    for idx, reduction_factor in enumerate(DEFAULT_REDUCTION_FACTORS):
        cum_npv = strategy3_results_cumulative_npv[reduction_factor]
        # Pad with 0 at year 0 for plotting from origin if needed
        plot_years = range(1, len(cum_npv) + 1) # Adjust length based on actual simulation duration
        ax_cum_npv.plot(plot_years, cum_npv, marker='.', linestyle='-', linewidth=2, markersize=6, color=colors[idx], label=f'S3 Cum. NPV (RF={reduction_factor:.1f})')

    ax_cum_npv.set_xlabel('Year', fontsize=12)
    ax_cum_npv.set_ylabel('Cumulative NPV ($)', fontsize=12)
    ax_cum_npv.set_title(f'Strategy 3 Cumulative NPV Evolution vs. Single-Period Outcomes', fontsize=12)
    ax_cum_npv.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Strategies / S3 RF")
    ax_cum_npv.grid(True)
    ax_cum_npv.ticklabel_format(style='plain', axis='y')
    ax_cum_npv.set_xticks(range(0, SIMULATION_YEARS + 1, max(1, SIMULATION_YEARS // 10)))

    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95]) # Adjust right margin for legend

    # --- Original Detailed Simulation Functions (Keep for reference/potential reuse) ---
    # def simulate_strategy_3(default_reduction_factor): ... (Original single-period version)
