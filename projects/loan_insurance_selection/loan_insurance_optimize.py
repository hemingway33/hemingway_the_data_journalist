# Sample Data Generation
import pandas as pd
import numpy as np

# Parameters
num_loans = 20
np.random.seed(240)

loan_data = pd.DataFrame({
    'LoanID': [f'L{i:03}' for i in range(num_loans)],
    'Amount': np.random.randint(50000, 500000, num_loans),
    'Rate': np.random.uniform(0.078, 0.15, num_loans), # Rates between 7.8% and 15%
    'PD': np.random.uniform(0.02, 0.18, num_loans)    # PD between 1% and 6%
})

# Ensure some rates need adjustment potentially
# Make a few rates slightly above 9% but potentially profitable to insure
high_rate_indices = np.random.choice(loan_data.index, 5, replace=False)
loan_data.loc[high_rate_indices, 'Rate'] = np.random.uniform(0.091, 0.11, 5)

# Ensure some rates are already low but maybe PD reduction is valuable
low_rate_indices = np.random.choice(loan_data.drop(high_rate_indices).index, 3, replace=False)
loan_data.loc[low_rate_indices, 'Rate'] = np.random.uniform(0.078, 0.085, 3)

# Ensure minimum rate reduction (0.5%) and minimum adjusted rate (7%) are possible
loan_data['Max_Possible_Reduction'] = loan_data['Rate'] - 0.07
loan_data = loan_data[loan_data['Max_Possible_Reduction'] >= 0.005].reset_index(drop=True) # Filter out loans that cannot meet constraints

print("Sample Loan Data:")
print(loan_data.to_string())

# Define model parameters
MAX_AVG_INSURED_RATE = 0.09
MIN_ADJUSTED_RATE = 0.07
INSURANCE_FEE = 0.005
MIN_RATE_REDUCTION = 0.005
PD_REDUCTION_BENEFIT = 0.024
EPSILON = 1e-7 # Small weight for secondary objective (maximize insured revenue)
# LAMBDA1 = 1e-3 # Optional penalty for number of insured loans (regularization)
# LAMBDA2 = 1e-1 # Optional penalty for total rate reduction (regularization)


# Pre-calculate values
loans = loan_data.to_dict('records')
loan_indices = range(len(loans))

for i in loan_indices:
    loans[i]['PD_adj'] = max(0, loans[i]['PD'] - PD_REDUCTION_BENEFIT)
    # Max reduction possible while respecting min adjusted rate
    loans[i]['Max_Reduction'] = loans[i]['Rate'] - MIN_ADJUSTED_RATE
    # Check feasibility: Max reduction must be >= min required reduction
    if loans[i]['Max_Reduction'] < MIN_RATE_REDUCTION:
         print(f"Warning: Loan {loans[i]['LoanID']} cannot be insured due to rate constraints. Rate={loans[i]['Rate']:.4f}, Max_Reduction={loans[i]['Max_Reduction']:.4f} < {MIN_RATE_REDUCTION}")
         # This loan should ideally be excluded or handled, but the constraint z_i <= M*x_i will handle it if M is calculated correctly.
         # Let's ensure M is calculated correctly:
         loans[i]['M_reduction'] = max(0, loans[i]['Rate'] - MIN_ADJUSTED_RATE) # Ensure M is non-negative
    else:
         loans[i]['M_reduction'] = loans[i]['Rate'] - MIN_ADJUSTED_RATE


# --- Use PuLP ---
import pulp

# Create the problem
prob = pulp.LpProblem("Loan_Insurance_Optimization", pulp.LpMaximize)

# --- Define Decision Variables ---
# x_i: 1 if loan i is insured, 0 otherwise
x = pulp.LpVariable.dicts("Insure", loan_indices, cat='Binary')

# z_i: Rate reduction for loan i if insured (continuous)
z = pulp.LpVariable.dicts("RateReduction", loan_indices, lowBound=0, cat='Continuous')

# --- Define Objective Function ---
# Maximize Sum [ x_i * L_i * (p_i^0 - p_i^adj - insurance_fee + epsilon * r_i^0) - z_i * L_i * (1 + epsilon) ]
# Optional regularization terms: - LAMBDA1 * Sum[x_i] - LAMBDA2 * Sum[z_i]
prob += pulp.lpSum(
    x[i] * loans[i]['Amount'] * (loans[i]['PD'] - loans[i]['PD_adj'] - INSURANCE_FEE + EPSILON * loans[i]['Rate'])
    - z[i] * loans[i]['Amount'] * (1 + EPSILON)
    # - LAMBDA1 * x[i] # Optional regularization
    # - LAMBDA2 * z[i] # Optional regularization
    for i in loan_indices
)

# --- Define Constraints ---

# Constraint 1: Min rate reduction if insured
for i in loan_indices:
    prob += z[i] >= MIN_RATE_REDUCTION * x[i], f"MinReduction_{i}"

# Constraint 2: Max rate reduction (to meet min adjusted rate) if insured
for i in loan_indices:
    # Ensure M is calculated correctly, using the pre-calculated 'M_reduction'
    M_i = loans[i]['M_reduction']
    prob += z[i] <= M_i * x[i], f"MaxReduction_{i}"
    # If M_i < MIN_RATE_REDUCTION, constraints 1 and 2 combined with x[i]=1 are infeasible for this loan, correctly preventing it from being insured.

# Constraint 3: Weighted average rate of insured loans <= MAX_AVG_INSURED_RATE
# Sum[ L_i * (r_i^0 - MAX_AVG_INSURED_RATE) * x_i - L_i * z_i ] <= 0
prob += pulp.lpSum(
    loans[i]['Amount'] * (loans[i]['Rate'] - MAX_AVG_INSURED_RATE) * x[i]
    - loans[i]['Amount'] * z[i]
    for i in loan_indices
) <= 0, "AvgInsuredRateLimit"

# --- Solve the Problem ---
print("\nSolving the MILP problem...")
# You might need to specify a solver path if pulp doesn't find one
# solver = pulp.PULP_CBC_CMD(path='path/to/cbc')
# prob.solve(solver)
prob.solve()

# --- Display Results ---
print(f"\nSolver Status: {pulp.LpStatus[prob.status]}")

if pulp.LpStatus[prob.status] == 'Optimal':
    print(f"Optimal Total Objective Value (Profit + epsilon*Revenue): {pulp.value(prob.objective):,.2f}")

    results = []
    total_amount_insured = 0
    total_insured_loan_weighted_rate_numerator = 0
    original_portfolio_profit = 0
    new_portfolio_profit = 0

    for i in loan_indices:
        loan = loans[i]
        original_profit_i = loan['Amount'] * (loan['Rate'] - loan['PD'])
        original_portfolio_profit += original_profit_i

        is_insured = (x[i].varValue > 0.5)
        reduction = z[i].varValue if is_insured else 0

        if is_insured:
            adjusted_rate = loan['Rate'] - reduction
            adjusted_pd = loan['PD_adj']
            profit_i = loan['Amount'] * (adjusted_rate - adjusted_pd - INSURANCE_FEE)
            new_portfolio_profit += profit_i

            total_amount_insured += loan['Amount']
            total_insured_loan_weighted_rate_numerator += loan['Amount'] * adjusted_rate

            results.append({
                'LoanID': loan['LoanID'],
                'Amount': loan['Amount'],
                'Original Rate': loan['Rate'],
                'Original PD': loan['PD'],
                'Insured': 'Yes',
                'Rate Reduction': reduction,
                'Adjusted Rate': adjusted_rate,
                'Adjusted PD': adjusted_pd
            })
        else:
            profit_i = original_profit_i
            new_portfolio_profit += profit_i
            # Keep non-insured loans in results for completeness if desired
            # results.append({
            #     'LoanID': loan['LoanID'],
            #     'Amount': loan['Amount'],
            #     'Original Rate': loan['Rate'],
            #     'Original PD': loan['PD'],
            #     'Insured': 'No',
            #     'Rate Reduction': 0,
            #     'Adjusted Rate': loan['Rate'],
            #     'Adjusted PD': loan['PD']
            # })


    results_df = pd.DataFrame(results)
    print("\nLoans Selected for Insurance and Adjustments:")
    if not results_df.empty:
        print(results_df.to_string(formatters={
            'Amount': '{:,.0f}'.format,
            'Original Rate': '{:.4f}'.format,
            'Original PD': '{:.4f}'.format,
            'Rate Reduction': '{:.4f}'.format,
            'Adjusted Rate': '{:.4f}'.format,
            'Adjusted PD': '{:.4f}'.format
        }))
    else:
        print("No loans were selected for insurance.")

    print(f"\nOriginal Portfolio Expected Profit: {original_portfolio_profit:,.2f}")
    print(f"Optimized Portfolio Expected Profit: {new_portfolio_profit:,.2f}")
    print(f"Increase in Expected Profit: {(new_portfolio_profit - original_portfolio_profit):,.2f}")

    if total_amount_insured > 0:
        avg_insured_rate = total_insured_loan_weighted_rate_numerator / total_amount_insured
        print(f"\nTotal Amount Insured: {total_amount_insured:,.0f}")
        print(f"Number of Loans Insured: {len(results_df)}")
        print(f"Weighted Average Interest Rate of Insured Loans: {avg_insured_rate:.4f}")
        if abs(avg_insured_rate - MAX_AVG_INSURED_RATE) < 1e-5:
             print(f"(Average rate is at the limit of {MAX_AVG_INSURED_RATE:.4f})")
        elif avg_insured_rate < MAX_AVG_INSURED_RATE:
             print(f"(Average rate is below the limit of {MAX_AVG_INSURED_RATE:.4f})")

    # Check Regularization effects if lambdas were used
    if not results_df.empty:
        total_reduction_amount = results_df['Rate Reduction'].sum()
        print(f"Sum of Rate Reductions (z_i): {total_reduction_amount:.4f}")


elif pulp.LpStatus[prob.status] == 'Infeasible':
    print("The problem is infeasible. Check constraints and data.")
    # Common reasons:
    # 1. A loan might have Rate < MIN_ADJUSTED_RATE + MIN_RATE_REDUCTION initially.
    # 2. Conflicting constraints.
else:
    print("Solver did not find an optimal solution.")




# --- Scenario 2: All Loans Insured ---
print("\n--- Scenario 2: All Loans MUST be Insured ---")

# Use the same data and parameters from the previous run
# loans, loan_indices, loan_data
# MAX_AVG_INSURED_RATE, MIN_ADJUSTED_RATE, INSURANCE_FEE, MIN_RATE_REDUCTION, PD_REDUCTION_BENEFIT

# 1. Feasibility Check
infeasible_all_insured = False
for i in loan_indices:
    loan = loans[i]
    if loan['Rate'] - MIN_RATE_REDUCTION < MIN_ADJUSTED_RATE:
        print(f"ERROR: Loan {loan['LoanID']} (Rate: {loan['Rate']:.4f}) cannot meet minimum reduction ({MIN_RATE_REDUCTION:.4f}) and minimum adjusted rate ({MIN_ADJUSTED_RATE:.4f}).")
        infeasible_all_insured = True
        break # Stop checking if one loan fails

if infeasible_all_insured:
    print("The 'All Insured' scenario is INFEASIBLE because at least one loan cannot satisfy the rate constraints.")
else:
    print("Initial feasibility check passed: All loans can individually meet the min reduction and min adjusted rate rules.")

    # Create the LP problem
    prob_all = pulp.LpProblem("Loan_Insurance_All_Insured", pulp.LpMinimize)

    # --- Define Decision Variables ---
    # z_i: Rate reduction for loan i (continuous)
    # Set bounds directly based on constraints
    z_all = {}
    for i in loan_indices:
        loan = loans[i]
        upper_bound_zi = loan['Rate'] - MIN_ADJUSTED_RATE
        z_all[i] = pulp.LpVariable(f"RateReduction_{i}", lowBound=MIN_RATE_REDUCTION, upBound=upper_bound_zi, cat='Continuous')

    # --- Define Objective Function ---
    # Minimize Sum [ L_i * z_i ]
    prob_all += pulp.lpSum(loans[i]['Amount'] * z_all[i] for i in loan_indices), "Total_Weighted_Reduction"

    # --- Define Constraints ---
    # Individual bounds are already set in variable definitions.

    # Constraint: Weighted average rate <= MAX_AVG_INSURED_RATE
    # Sum[ L_i * z_i ] >= Sum[ L_i * (r_i^0 - MAX_AVG_INSURED_RATE) ]
    required_total_weighted_reduction = sum(
        loans[i]['Amount'] * (loans[i]['Rate'] - MAX_AVG_INSURED_RATE)
        for i in loan_indices
    )
    print(f"Required Total Weighted Reduction (Sum[L_i*z_i]) >= {required_total_weighted_reduction:,.2f}")

    prob_all += pulp.lpSum(
        loans[i]['Amount'] * z_all[i] for i in loan_indices
    ) >= required_total_weighted_reduction, "AvgInsuredRateLimit_All"

    # --- Solve the Problem ---
    print("\nSolving the LP problem for 'All Insured' scenario...")
    prob_all.solve()

    # --- Display Results ---
    print(f"\nSolver Status (All Insured): {pulp.LpStatus[prob_all.status]}")

    if pulp.LpStatus[prob_all.status] == 'Optimal':
        print(f"Minimum Total Weighted Reduction Objective: {pulp.value(prob_all.objective):,.2f}")

        results_all = []
        total_amount_insured_all = 0
        total_insured_loan_weighted_rate_numerator_all = 0
        new_portfolio_profit_all = 0

        for i in loan_indices:
            loan = loans[i]
            reduction = z_all[i].varValue
            adjusted_rate = loan['Rate'] - reduction
            adjusted_pd = loan['PD_adj'] # PD reduction benefit applies to all

            profit_i = loan['Amount'] * (adjusted_rate - adjusted_pd - INSURANCE_FEE)
            new_portfolio_profit_all += profit_i

            total_amount_insured_all += loan['Amount']
            total_insured_loan_weighted_rate_numerator_all += loan['Amount'] * adjusted_rate

            results_all.append({
                'LoanID': loan['LoanID'],
                'Amount': loan['Amount'],
                'Original Rate': loan['Rate'],
                'Original PD': loan['PD'],
                'Insured': 'Yes (Forced)',
                'Rate Reduction': reduction,
                'Adjusted Rate': adjusted_rate,
                'Adjusted PD': adjusted_pd
            })

        results_all_df = pd.DataFrame(results_all)
        print("\nAdjustments for 'All Insured' Scenario:")
        print(results_all_df.to_string(formatters={
            'Amount': '{:,.0f}'.format,
            'Original Rate': '{:.4f}'.format,
            'Original PD': '{:.4f}'.format,
            'Rate Reduction': '{:.4f}'.format,
            'Adjusted Rate': '{:.4f}'.format,
            'Adjusted PD': '{:.4f}'.format
        }))

        # Use original_portfolio_profit calculated in the first scenario
        print(f"\nOriginal Portfolio Expected Profit: {original_portfolio_profit:,.2f}") # Assuming this variable exists from previous run
        print(f"Optimized Portfolio Expected Profit (All Insured): {new_portfolio_profit_all:,.2f}")
        print(f"Increase in Expected Profit (All Insured): {(new_portfolio_profit_all - original_portfolio_profit):,.2f}")

        if total_amount_insured_all > 0:
            avg_insured_rate_all = total_insured_loan_weighted_rate_numerator_all / total_amount_insured_all
            print(f"\nTotal Amount Insured: {total_amount_insured_all:,.0f}")
            print(f"Number of Loans Insured: {len(results_all_df)}")
            print(f"Weighted Average Interest Rate of Insured Loans: {avg_insured_rate_all:.4f}")
            if abs(avg_insured_rate_all - MAX_AVG_INSURED_RATE) < 1e-5:
                 print(f"(Average rate is at the limit of {MAX_AVG_INSURED_RATE:.4f})")
            elif avg_insured_rate_all < MAX_AVG_INSURED_RATE:
                 print(f"(Average rate is below the limit of {MAX_AVG_INSURED_RATE:.4f})")
            else:
                 print(f"WARNING: Average rate {avg_insured_rate_all:.4f} EXCEEDS the limit {MAX_AVG_INSURED_RATE:.4f}. Check constraints/solver tolerance.")


            total_reduction_amount_all = results_all_df['Rate Reduction'].sum()
            print(f"Sum of Rate Reductions (z_i): {total_reduction_amount_all:.4f}")

    elif pulp.LpStatus[prob_all.status] == 'Infeasible':
        print("The 'All Insured' problem is infeasible.")
        print("This could happen if:")
        print(f"  1. The sum of minimum required reductions (Sum[L_i * {MIN_RATE_REDUCTION}]) is already too high.")
        print(f"  2. The total weighted reduction needed ({required_total_weighted_reduction:,.2f}) cannot be achieved within the allowed max reductions (z_i <= M_i).")
        # Check the bounds sum vs required reduction
        max_possible_total_weighted_reduction = sum(loans[i]['Amount'] * (loans[i]['Rate'] - MIN_ADJUSTED_RATE) for i in loan_indices)
        min_required_total_weighted_reduction_by_min_z = sum(loans[i]['Amount'] * MIN_RATE_REDUCTION for i in loan_indices)
        print(f"  - Minimum possible Sum[L_i*z_i] (based on z_i >= {MIN_RATE_REDUCTION}): {min_required_total_weighted_reduction_by_min_z:,.2f}")
        print(f"  - Maximum possible Sum[L_i*z_i] (based on z_i <= M_i): {max_possible_total_weighted_reduction:,.2f}")
        print(f"  - Required Sum[L_i*z_i] >= {required_total_weighted_reduction:,.2f}")


    else:
        print("Solver did not find an optimal solution for the 'All Insured' scenario.")

# --- Comparison Summary ---
print("\n--- Comparison Summary ---")
print(f"Selective Insurance Profit: {new_portfolio_profit:,.2f}" if 'new_portfolio_profit' in locals() else "Selective Insurance: Not run or failed")
print(f"'All Insured' Profit:       {new_portfolio_profit_all:,.2f}" if 'new_portfolio_profit_all' in locals() and pulp.LpStatus[prob_all.status] == 'Optimal' else "'All Insured': Infeasible or failed")

if 'new_portfolio_profit' in locals() and 'new_portfolio_profit_all' in locals() and pulp.LpStatus[prob_all.status] == 'Optimal':
    profit_diff = new_portfolio_profit - new_portfolio_profit_all
    print(f"\nDifference (Selective - All Insured): {profit_diff:,.2f}")
    if profit_diff > 0:
        print("Selective insurance provides higher expected profit.")
    elif profit_diff < 0:
        print("'All Insured' provides higher expected profit (this is unusual unless selective model had issues or very few loans).")
    else:
        print("Both scenarios yield the same expected profit.")

    # Compare number of adjustments
    num_selective = len(results_df) if 'results_df' in locals() and not results_df.empty else 0
    num_all = len(results_all_df) if 'results_all_df' in locals() and not results_all_df.empty else 0
    print(f"\nNumber of Loans Insured/Adjusted:")
    print(f"Selective Insurance: {num_selective}")
    print(f"'All Insured':       {num_all}")