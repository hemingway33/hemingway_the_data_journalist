import pandas as pd
import numpy as np

# --- Configuration ---

def get_default_parameters():
    """Returns a dictionary containing the default parameters from the spreadsheet."""
    params = {
        # Single Account Model Params
        'initial_balance_new_cust': 13.00,  # 新户平均余额 (万)
        'balance_growth_rate': 0.09,       # 余额增长率 (%)
        'avg_interest_rate_new_cust': 0.144, # 新户收息率 (平均) (%)
        'interest_rate_growth': -0.006,    # 收息增长率 (%) - Assuming additive change per year
        'funding_cost_rate': 0.035,      # 资金成本 (%)

        # Removed hardcoded gross income, will calculate dynamically
        # 'gross_income_year': {
        #     1: 1.80, 2: 1.97, 3: 2.15, 4: 2.32, 5: 2.50, 6: 2.71
        # },

        'loss_rate_year': { # 预期损失率假设 (%)
            1: 0.11, 2: 0.09, 3: 0.07, 4: 0.08, 5: 0.09, 6: 0.09
        },
        'marketing_cost_year': { # 市场营销费 (万元)
            1: 0.30, 2: 0.10, 3: 0.06, 4: 0.04, 5: 0.02, 6: 0.01
        },
        'account_mgt_cost_year': { # 账户运营费 (万元)
            1: 0.10, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.13, 6: 0.13
        },
        'acquisition_cost_year': { # 催收成本 (万元) - Assuming "催收成本" means Acquisition/Collection
             1: 0.04, 2: 0.04, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.05
        },
         'core_platform_cost_year': { # 核销成本 (万元) - Assuming "核销成本" means Core Platform/Write-off Cost
             1: 0.48, 2: 1.38, 3: 1.21, 4: 1.17, 5: 1.45, 6: 1.70
         },

        # Overall Product Model Params
        'new_customers_per_year': 10000, # 每年新增新户 (户)
        'retention_rate': 0.75,          # 跨年综合留存率 (%)
        'simulation_years': 6            # How many years to simulate the portfolio
    }
    return params

# --- Calculation Logic ---

def calculate_single_account_profit(params):
    """Calculates the profit/loss per account for each year dynamically."""
    max_years = params['simulation_years']
    account_profit = pd.DataFrame(index=pd.RangeIndex(start=1, stop=max_years + 1, name='Account Year'))

    # Calculate balance trajectory dynamically
    balances = []
    current_balance = params['initial_balance_new_cust']
    for year in range(1, max_years + 1):
        balances.append(current_balance)
        # Apply growth rate for the *next* year's starting balance
        current_balance *= (1 + params['balance_growth_rate'])
    account_profit['Balance (万)'] = balances

    # Calculate interest rate trajectory dynamically (additive growth)
    interest_rates = []
    current_rate = params['avg_interest_rate_new_cust']
    for year in range(1, max_years + 1):
        interest_rates.append(current_rate)
        # Apply growth for the next year's rate
        current_rate += params['interest_rate_growth'] # Additive change
        # Ensure rate doesn't go below zero (or some floor)
        current_rate = max(0, current_rate)
    account_profit['Interest Rate (%)'] = interest_rates

    # Get Loss Rate for calculation
    account_profit['Loss Rate (%)'] = [params['loss_rate_year'].get(y, 0) for y in account_profit.index]

    # Calculate Interest Earning Balance
    # Applying the "1/3 rule": Interest Earning Balance = Balance * (1 - Loss Rate / 3)
    account_profit['Interest Earning Balance (万)'] = account_profit['Balance (万)'] * (1 - account_profit['Loss Rate (%)'] / 3.0)

    # Calculate Gross Income dynamically
    account_profit['Gross Income (万)'] = account_profit['Interest Earning Balance (万)'] * account_profit['Interest Rate (%)']

    # Calculate Funding Cost dynamically (assuming cost applies to total balance)
    account_profit['Funding Cost (万)'] = account_profit['Balance (万)'] * params['funding_cost_rate']

    # Calculate Net Income
    account_profit['Net Income (万)'] = account_profit['Gross Income (万)'] - account_profit['Funding Cost (万)']

    # Calculate Operating Expenses dynamically from params
    account_profit['Marketing Cost (万)'] = [params['marketing_cost_year'].get(y, 0) for y in account_profit.index]
    account_profit['Account Mgt Cost (万)'] = [params['account_mgt_cost_year'].get(y, 0) for y in account_profit.index]
    account_profit['Acquisition Cost (万)'] = [params['acquisition_cost_year'].get(y, 0) for y in account_profit.index]
    account_profit['Core Platform Cost (万)'] = [params['core_platform_cost_year'].get(y, 0) for y in account_profit.index]
    account_profit['Total OpEx (万)'] = (
        account_profit['Marketing Cost (万)'] +
        account_profit['Account Mgt Cost (万)'] +
        account_profit['Acquisition Cost (万)'] +
        account_profit['Core Platform Cost (万)']
    )

    # Calculate Expected Loss dynamically
    # Loss Rate column already added above
    account_profit['Expected Loss (万)'] = account_profit['Balance (万)'] * account_profit['Loss Rate (%)']

    # Calculate Final Net Profit/Loss
    account_profit['Net Profit/Loss (万)'] = account_profit['Net Income (万)'] - account_profit['Total OpEx (万)'] - account_profit['Expected Loss (万)']

    # Select and order columns for clarity
    output_cols = [
        'Balance (万)',
        'Interest Rate (%)',
        'Loss Rate (%)',
        'Interest Earning Balance (万)',
        'Gross Income (万)',
        'Funding Cost (万)',
        'Net Income (万)',
        'Marketing Cost (万)',
        'Account Mgt Cost (万)',
        'Acquisition Cost (万)',
        'Core Platform Cost (万)',
        'Total OpEx (万)',
        'Expected Loss (万)',
        'Net Profit/Loss (万)'
    ]
    return account_profit[output_cols]


def simulate_portfolio_profit(params, account_profit_df):
    """Simulates the overall portfolio profit over the specified number of years."""
    n_years = params['simulation_years']
    new_cust = params['new_customers_per_year']
    retention = params['retention_rate']

    # Track cohorts
    cohorts = {} # Key: year cohort started, Value: Dataframe of cohort size over time

    portfolio_summary = pd.DataFrame(index=range(1, n_years + 1))
    portfolio_summary.index.name = 'Business Year'

    total_profit_per_year = []
    total_customers_per_year = []
    profit_details = {} # Store profit breakdown by cohort age

    # Pre-fetch the profit/loss series for quick lookup
    account_pl = account_profit_df['Net Profit/Loss (万)']

    for year in range(1, n_years + 1):
        current_year_total_profit = 0
        current_year_total_customers = 0

        # Add new cohort
        cohorts[year] = pd.DataFrame({'customers': [new_cust]}, index=[1]) # Year 1 for this cohort
        current_year_total_customers += new_cust
        # Use .get(1, 0) for safety if account_pl index doesn't start at 1 or is missing
        profit_from_new_cohort = new_cust * account_pl.get(1, 0)
        current_year_total_profit += profit_from_new_cohort
        profit_details[(year, 1)] = profit_from_new_cohort # (Business Year, Cohort Age) -> Profit

        # Update existing cohorts
        for cohort_start_year, cohort_data in cohorts.items():
            if cohort_start_year == year: # Skip the cohort just added
                continue

            last_age = cohort_data.index.max()
            current_age = year - cohort_start_year + 1

            if current_age <= n_years: # Only track within simulation period
                # Apply retention
                retained_customers = cohort_data.loc[last_age, 'customers'] * retention
                cohort_data.loc[current_age] = [retained_customers] # Add new row for current age
                current_year_total_customers += retained_customers

                # Calculate profit from this aged cohort
                # Use .get(current_age, 0) for safety
                profit_per_account = account_pl.get(current_age, 0)
                profit_from_cohort = retained_customers * profit_per_account
                current_year_total_profit += profit_from_cohort
                profit_details[(year, current_age)] = profit_from_cohort

        total_customers_per_year.append(current_year_total_customers)
        total_profit_per_year.append(current_year_total_profit)

    portfolio_summary['Total Customers'] = total_customers_per_year
    portfolio_summary['Total Annual Profit (万)'] = total_profit_per_year
    portfolio_summary['Cumulative Profit (万)'] = portfolio_summary['Total Annual Profit (万)'].cumsum()

    # Optional: create detailed profit breakdown dataframe
    profit_breakdown = pd.DataFrame(index=portfolio_summary.index)
    for year in portfolio_summary.index:
        for age in range(1, year + 1):
             profit_breakdown.loc[year, f'Age {age} Cohort Profit (万)'] = profit_details.get((year, age), 0)
    profit_breakdown = profit_breakdown.fillna(0)


    return portfolio_summary, profit_breakdown, cohorts

# --- Main Execution / Simulation Setup ---

# Function to run a single simulation scenario
# def run_single_simulation(params):
#     """Runs account and portfolio calculation for a given set of parameters."""
#     try:
#         # Calculate account profit dynamically based on potentially randomized params
#         account_profit = calculate_single_account_profit(params)
#         # Simulate portfolio using these dynamic params and account profit
#         portfolio_summary, _, _ = simulate_portfolio_profit(params, account_profit)
#         return portfolio_summary
#     except Exception as e:
#         print(f"Error during simulation run: {e}")
#         return None


# Function to generate randomized parameters for a simulation run
def get_randomized_parameters(base_params):
    """Generates a new parameter set by randomizing specific base parameters."""
    params = base_params.copy() # Start with a copy of base params

    # --- Define Parameter Variations ---
    # Retention Rate: Uniform distribution between 65% and 85%
    params['retention_rate'] = np.random.uniform(0.65, 0.85)

    # Marketing Cost Multiplier: Normal distribution, mean 1.0, std dev 0.2 (min 0.5)
    marketing_multiplier = max(0.5, np.random.normal(loc=1.0, scale=0.2))
    params['marketing_cost_year'] = {
        year: cost * marketing_multiplier
        for year, cost in base_params['marketing_cost_year'].items()
    }

    # Loss Rate Multiplier: Normal distribution, mean 1.0, std dev 0.3 (min 0.3)
    loss_multiplier = max(0.3, np.random.normal(loc=1.0, scale=0.3))
    params['loss_rate_year'] = {
        year: rate * loss_multiplier
        for year, rate in base_params['loss_rate_year'].items()
    }
    # --- End Parameter Variations ---

    return params


if __name__ == "__main__":
    # 1. Get default parameters (base for randomization)
    default_params = get_default_parameters()
    print("--- Default Parameters (Base for Simulation) ---")
    print(f"Initial Balance: {default_params['initial_balance_new_cust']:.2f} 万")
    print(f"Initial Interest Rate: {default_params['avg_interest_rate_new_cust']:.1%}")
    print(f"Interest Rate Growth (Annual Additive): {default_params['interest_rate_growth']:.1%}")
    print(f"Funding Cost Rate: {default_params['funding_cost_rate']:.1%}")
    print(f"Balance Growth Rate: {default_params['balance_growth_rate']:.1%}")
    print(f"Default Retention Rate: {default_params['retention_rate']:.1%}")
    print("-" * 25)

    # 2. Calculate single account profit trajectory (using DEFAULTS and FULLY DYNAMIC calculation)
    account_profit_default_dynamic = calculate_single_account_profit(default_params)
    print("\n--- Single Account Profit/Loss (Default Params, Fully Dynamic Calculation) ---")
    print("Note: Uses dynamic income (Balance*(1-LossRate)*InterestRate) & loss calc.")
    print("      May differ significantly from original sheet profit values.")
    print(account_profit_default_dynamic[['Balance (万)', 'Interest Rate (%)', 'Gross Income (万)', 'Expected Loss (万)', 'Net Profit/Loss (万)']].round(2))
    print("-" * 25)

    # 3. Simulate portfolio profit using default parameters (and dynamic account profit calc)
    portfolio_summary_default_dynamic, _, _ = simulate_portfolio_profit(default_params, account_profit_default_dynamic)
    print("\n--- Portfolio Simulation Summary (Default Params, Fully Dynamic Calculation) ---")
    print(portfolio_summary_default_dynamic.round(2))
    print("-" * 25)


    # --- Monte Carlo Simulation ---
    print("\n--- Running Monte Carlo Simulation ---")
    N_SIMULATIONS = 1000  # Number of simulation runs
    results_summary = []  # Store summary results (like initial loss status)
    # all_portfolio_summaries = [] # Optional: Store full summary dataframes if memory allows

    initial_loss_scenario_count = 0 # Count scenarios with loss in Year 1 or 2

    for i in range(N_SIMULATIONS):
        # Generate randomized parameters for this run
        random_params = get_randomized_parameters(default_params)

        # Calculate account profit dynamically using the random parameters for this run
        current_account_profit = calculate_single_account_profit(random_params)

        # Run the portfolio simulation with these parameters and the dynamic account profit
        portfolio_summary = simulate_portfolio_profit(random_params, current_account_profit)[0] # Get summary dataframe

        if portfolio_summary is not None:
            # Store key results or the full summary
            # all_portfolio_summaries.append(portfolio_summary)

            # Check for initial loss (Year 1 or Year 2 Profit < 0)
            year_1_profit = portfolio_summary.loc[1, 'Total Annual Profit (万)']
            year_2_profit = portfolio_summary.loc[2, 'Total Annual Profit (万)'] if 2 in portfolio_summary.index else 0

            is_initial_loss = (year_1_profit < 0) or (year_2_profit < 0)
            if is_initial_loss:
                initial_loss_scenario_count += 1

            # Extract multipliers for reporting (using Year 1 as example)
            loss_mult = random_params['loss_rate_year'].get(1,0) / default_params['loss_rate_year'].get(1,1)
            mktg_mult = random_params['marketing_cost_year'].get(1,0) / default_params['marketing_cost_year'].get(1,1)

            results_summary.append({
                'run': i + 1,
                'retention_rate': random_params['retention_rate'],
                'loss_multiplier': loss_mult,
                'marketing_multiplier': mktg_mult,
                'year_1_profit': year_1_profit,
                'year_2_profit': year_2_profit,
                'year_6_profit': portfolio_summary.loc[6, 'Total Annual Profit (万)'] if 6 in portfolio_summary.index else 0,
                'initial_loss': is_initial_loss
            })

        # Optional: Print progress
        if (i + 1) % 100 == 0:
            print(f"Completed simulation {i+1}/{N_SIMULATIONS}")

    print(f"\n--- Simulation Results ({N_SIMULATIONS} runs) ---")

    if not results_summary:
        print("No simulation results generated.")
    else:
        results_df = pd.DataFrame(results_summary)

        # Calculate probability of initial loss
        prob_initial_loss = initial_loss_scenario_count / len(results_df)
        print(f"Probability of Negative Annual Profit in Year 1 or 2: {prob_initial_loss:.2%}")

        # Analyze scenarios with initial loss
        initial_loss_scenarios = results_df[results_df['initial_loss'] == True]
        if not initial_loss_scenarios.empty:
            print("\n--- Analysis of Scenarios with Initial Loss ---")
            print(f"Average Retention Rate in Loss Scenarios: {initial_loss_scenarios['retention_rate'].mean():.1%}")
            print(f"Average Loss Multiplier in Loss Scenarios: {initial_loss_scenarios['loss_multiplier'].mean():.2f}x")
            print(f"Average Marketing Multiplier in Loss Scenarios: {initial_loss_scenarios['marketing_multiplier'].mean():.2f}x")
        else:
            print("\nNo scenarios resulted in an initial loss under these dynamics.")

        # Analyze overall profit distribution
        print("\n--- Overall Profit Analysis --- ")
        print(f"Average Year 1 Profit: {results_df['year_1_profit'].mean():.2f} 万")
        print(f"Average Year 6 Profit: {results_df['year_6_profit'].mean():.2f} 万")
        print(f"Std Dev Year 6 Profit: {results_df['year_6_profit'].std():.2f} 万")


    print("\nSimulation complete. Check results above.")
    # print("Consider further analysis on 'results_df' or 'all_portfolio_summaries'.")
