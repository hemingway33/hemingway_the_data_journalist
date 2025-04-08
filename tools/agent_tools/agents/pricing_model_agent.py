"""Agent responsible for applying financial pricing models."""

import yfinance as yf
import pandas as pd

def run_pricing_model(state):
    """Runs the pricing model agent node using basic yfinance data."""
    print("--- Running Pricing Model Agent ---")
    topic = state["topic"]
    research_data = state.get("research_info", []) # Use research data if needed in future

    try:
        # Fetch stock data using yfinance
        stock = yf.Ticker(topic)
        info = stock.info

        # Basic data points for a simple model
        current_price = info.get('currentPrice')
        forward_pe = info.get('forwardPE')
        forward_eps = info.get('forwardEps')
        beta = info.get('beta', 1.0) # Default beta to 1 if not available
        debt_to_equity = info.get('debtToEquity')

        pricing_analysis = f"Pricing Model Analysis for {topic}:\n"
        pricing_analysis += f"- Current Price: ${current_price:.2f}\n"

        # Very Simple P/E based estimation (Conceptual - NOT FINANCIAL ADVICE)
        if forward_pe and forward_eps:
            estimated_value_pe = forward_pe * forward_eps
            pricing_analysis += f"- Forward P/E: {forward_pe:.2f}\n"
            pricing_analysis += f"- Forward EPS: ${forward_eps:.2f}\n"
            pricing_analysis += f"- Simple P/E Estimated Value: ${estimated_value_pe:.2f}\n"
            if current_price:
                 potential = ((estimated_value_pe / current_price) - 1) * 100
                 pricing_analysis += f"- Potential vs Current Price (P/E based): {potential:.2f}%\n"
        else:
            pricing_analysis += "- P/E or EPS data not available for simple estimation.\n"

        # Include other relevant metrics
        pricing_analysis += f"- Beta (Market Risk): {beta:.2f}\n"
        if debt_to_equity:
             pricing_analysis += f"- Debt-to-Equity: {debt_to_equity / 100:.2f} (as reported by source)\n"
        else:
             pricing_analysis += "- Debt-to-Equity data not available.\n"

        analysis_summary = f"Based on available yfinance data, a simple P/E analysis suggests an estimated value of approximately ${estimated_value_pe:.2f}. Key metrics like Beta and Debt-to-Equity provide context on risk."

    except Exception as e:
        print(f"Error fetching or processing data for {topic}: {e}")
        pricing_analysis = f"Could not perform pricing analysis for {topic} due to error: {e}"
        analysis_summary = "Pricing analysis could not be completed."

    state['messages'].append(f"Pricing Model Agent: Ran models for {topic}.")
    state['analysis'] = pricing_analysis # Store the detailed analysis string
    # Optionally add a summary to messages or keep it within analysis
    state['messages'].append(f"Pricing Model Summary: {analysis_summary}")
    return state 