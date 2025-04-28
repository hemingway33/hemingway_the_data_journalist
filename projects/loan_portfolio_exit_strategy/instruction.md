# Loan Portfolio Exit Strategy Simulation

## Project Goal

This simulation analyzes and compares different exit strategies for a portfolio of balloon payment loans over a multi-year horizon. The goal is to evaluate the Net Present Value (NPV) and Total Absolute Loss under various scenarios, considering factors like customer churn, funding costs, default rates, recovery rates, and specific strategy interventions.

## Portfolio & Baseline Assumptions

*   **Initial Portfolio:** 10,000 loans, $100,000 average balance per loan.
*   **Loan Type:** Balloon payment (interest paid monthly, principal at end of 12 months).
*   **Interest Rate:** 12% APR (1% monthly).
*   **Funding Cost:** 3% annual rate.
*   **Risk Bins:** 8 bins with increasing Predicted Default (PD) rates (2% to 35%). Bins 7 and 8 are considered high-risk.
*   **Standard Recovery:** 30% recovery on defaulted principal for non-treated loans.
*   **Simulation Horizon:** 3 years.
*   **Churn:** 30% annual churn rate (loans paid off/removed).
*   **Discount Rate:** 3% (equal to funding cost).

## Simulated Strategies

### Baseline (Standard Operation)

*   Loans run through their standard 12-month balloon term.
*   At the end of the term, loans either default (based on their bin's PD rate) or pay off the principal.
*   Defaulted loans recover principal based on the `STANDARD_RECOVERY_RATE` (30%).
*   Surviving loans (those that pay off) leave the portfolio.
*   Annual churn is applied at the start of each year.

### Strategy 1: Recall High-Risk Loans (断贷)

*   Same as Baseline for low-risk loans.
*   **High-Risk Loans (Bins 7 & 8):** These loans are recalled/terminated at the end of their initial 12-month term. The outcome depends on a specified `RECOVERY_RATES_STRAT1` (varied from 10% to 50% in the simulation). This means the lender forces repayment or liquidation, achieving a recovery rate different from the standard one.
*   The simulation tests the impact of different recovery rates achieved through this recall process.

### Strategy 2: Mandatory Extension (24m Amortization)

*   This strategy applies a mandatory 24-month extension with changes, simulated over a 2-year horizon (since the extension defines the period).
*   **Low-Risk Loans:** Extended for 24 months on an *amortizing* basis (principal and interest payments). Survival/default is based on the original `annual_pd`, modeled using a *piecewise exponential survival* model (higher default hazard in the first 6 months, ratio defined by `HAZARD_RATE_RATIO`).
*   **High-Risk Loans:**
    *   A small percentage (`REFUSAL_PROBABILITY`, 5%) refuse the extension and follow the Baseline default/payoff logic based on their original PD.
    *   The rest accept the extension:
        *   Receive an immediate principal haircut (`PRINCIPAL_REDUCTION_FACTOR`, 10%).
        *   Enter a 24-month *amortizing* repayment plan on the *reduced* principal.
        *   Interest rate is increased by `EXTENSION_RATE_INCREASE` (5%).
        *   Their annual PD rate is *reduced* by `DEFAULT_REDUCTION_FACTORS` (varied from 20% to 60% reduction).
        *   Survival/default during the 24 months is modeled using the *reduced PD* and the *piecewise exponential survival* model.
*   The simulation tests the impact of different default reduction factors achieved through this intervention.

### Strategy 3: Selective Extension (Indefinite Balloon)

*   Same as Baseline for low-risk loans.
*   **High-Risk Loans:**
    *   A small percentage (`REFUSAL_PROBABILITY`, 5%) refuse extension and follow Baseline logic.
    *   The rest accept the extension:
        *   Receive an immediate principal haircut (`PRINCIPAL_REDUCTION_FACTOR`, 10%).
        *   Enter a *new 12-month balloon term* on the *reduced* principal.
        *   Interest rate is increased by `EXTENSION_RATE_INCREASE` (5%).
        *   Their annual PD rate is *reduced* by `DEFAULT_REDUCTION_FACTORS` (varied from 20% to 60% reduction).
        *   At the end of this 12-month extension, they either default (based on the *reduced PD*) or pay off the reduced principal.
        *   If they pay off, they *can potentially repeat the process* in subsequent years (receive another haircut, another 12m balloon extension with the reduced PD) as long as the simulation runs. This allows for modeling an indefinite extension scenario.
*   The simulation tests the impact of different default reduction factors on this rolling extension strategy. 