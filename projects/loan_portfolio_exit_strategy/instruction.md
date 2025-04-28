# Simulation Instructions Summary

## Initial Goal
Simulate and compare loan portfolio exit strategies for a high-loss portfolio. Visualize results for profit maximization or loss minimization.

## Portfolio & Base Parameters
- **Loans:** 10,000
- **Average Balance:** $100,000
- **Structure:** 12-month balloon (11 months interest-only, principal payoff month 12)
- **Interest Rate:** 12% APR
- **PD Bins:** 8 bins with increasing Predicted Default rates (`PD_RATES`).
- **High-Risk Bins:** Indices 6 and 7.
- **Standard Recovery Rate:** 30%.
- **Funding Cost:** 3% annual rate.
- **Discount Rate:** 3% (same as funding cost).

## Simulation Time Horizon & Churn
- **Simulation Years:** 3 years (used for S1 and S3 multi-year horizon).
- **Churn Rate:** 30% annually, applied at the start of each year to the remaining portfolio in multi-year simulations.

## Baseline Strategy (S0)
- **Type:** Single-period calculation.
- **Logic:** No intervention. Original 12-month balloon loan structure.
- **Output:** 1-year Net P&L After Funding, 1-year Absolute Loss.

## Strategy 1: Recall
- **Type:** Multi-year simulation (over `SIMULATION_YEARS`).
- **Logic:**
    - Applied each year to loans currently classified in high-risk bins.
    - Loans are recalled at the end of their 12-month term.
    - Non-defaults pay off (earn 11 months interest).
    - Defaults occur based on bin PD.
    - **Parameter:** Recovery Rate (`RECOVERY_RATES_STRAT1`) applied specifically to *high-risk* defaults.
    - Non-high-risk bin defaults use `STANDARD_RECOVERY_RATE`.
- **Output:** NPV over `SIMULATION_YEARS`, Total Absolute Loss over `SIMULATION_YEARS`.

## Strategy 2: Mandatory 24m Amortization w/ High-Risk Treatment
- **Type:** Multi-year simulation (runs for exactly 2 years).
- **Logic:**
    - Applies after initial churn in Year 1.
    - **Rate Increase:** Interest rate increased by 5% absolute (`EXTENSION_RATE_INCREASE`) for *all* loans entering the 24m amortization.
    - **Low-Risk Loans:** Convert directly to a 24-month amortizing structure at the increased rate. Total 24m P&L/Loss calculated upfront and allocated 50/50 to Year 1 and Year 2.
    - **High-Risk Loans:**
        - **Refusal:** 5% probability (`REFUSAL_PROBABILITY`) refuse extension and follow Baseline default logic for Year 1.
        - **Acceptance (95%):**
            - **Haircut:** 10% immediate principal reduction (`PRINCIPAL_REDUCTION_FACTOR`). Haircut loss booked in Year 1.
            - **PD Reduction:** Parameter `DEFAULT_REDUCTION_FACTORS` applied to reduce the original bin PD for the 24m term.
            - **Extension:** Enter 24m amortization on *reduced* principal at the *increased* rate.
            - Total 24m P&L/Loss (including haircut impact) calculated upfront and allocated 50/50 to Year 1 and Year 2.
    - Portfolio fully exits after Year 2.
- **Output:** 2-year NPV, 2-year Total Absolute Loss, 2-year Cumulative NPV list.

## Strategy 3: Repeated 12m Balloon Extension w/ Haircut
- **Type:** Multi-year simulation (over `SIMULATION_YEARS`).
- **Logic:**
    - Applied each year only to loans currently classified in high-risk bins.
    - **Refusal:** 5% probability (`REFUSAL_PROBABILITY`) refuse extension and follow Baseline default logic for that year.
    - **Acceptance (95%):**
        - **Haircut:** 10% immediate principal reduction (`PRINCIPAL_REDUCTION_FACTOR`).
        - **Rate Increase:** Interest rate increased by 5% absolute (`EXTENSION_RATE_INCREASE`).
        - **Extension:** New 12-month balloon loan on the *reduced* principal at the *increased* rate.
        - **PD Reduction:** Parameter `DEFAULT_REDUCTION_FACTORS` applied to reduce the original bin PD for this 12-month extension term.
        - **Default Timing (Split):** 1/3 of total defaults occur mid-term (months 1-11, loss only), 2/3 occur at end-term (month 12 payoff, interest collected before loss).
        - **Continuation:** Survivors continue to the next year with the reduced balance, subject to churn and re-evaluation (may remain high-risk and repeat S3).
- **Output:** NPV over `SIMULATION_YEARS`, Total Absolute Loss over `SIMULATION_YEARS`, Yearly Cumulative NPV list, Yearly Balance list.

## Comparisons & Outputs
- **Metrics:** Net P&L After Funding (Baseline), NPV (S1, S2, S3), Absolute Loss (Baseline), Total Absolute Loss (S1, S2, S3 horizon).
- **Plot 1 (NPV):** Compare S1 NPV, S2 NPV, S3 NPV vs. parameters, show Baseline P&L AF reference.
- **Plot 2 (Loss):** Compare S1 Total Loss, S2 Total Loss, S3 Total Loss vs. parameters, show Baseline Abs Loss reference.
- **Plot 3 (Cumulative NPV):** Show cumulative NPV evolution for S2 (2 years) and S3 (full horizon) over time. Show Baseline marker (Yr 1 P&L) and S1 range bar (Final NPV at end year) for context.
- **Plot 4 (S3 Balance):** Show portfolio balance evolution over time for S3 under different reduction factors.
- **CSV Exports:**
    - `results_single_period_summary.csv`: Baseline metrics.
    - `results_s3_yearly_pnl_af.csv`: Strategy 3 yearly Net P&L AF, columns by reduction factor.
    - `results_s3_yearly_loss.csv`: Strategy 3 yearly Absolute Loss, columns by reduction factor.

*(Note: Some parameter values and specific plot representations were adjusted during the process based on clarifications and iterative development.)* 