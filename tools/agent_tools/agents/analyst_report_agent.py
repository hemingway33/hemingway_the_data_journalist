"""Agent responsible for fetching and understanding analyst reports/recommendations."""

import yfinance as yf
import pandas as pd

def run_analyst_report_analysis(state):
    """Runs the analyst report agent node using yfinance recommendation data."""
    print("--- Running Analyst Report Agent ---")
    topic = state["topic"]

    try:
        stock = yf.Ticker(topic)
        recs = stock.recommendations_summary
        info = stock.info # For target price if available

        analyst_summary = f"Analyst Recommendation Summary for {topic}:\n"

        if not recs.empty:
            latest_recs = recs.iloc[-1] # Get the most recent summary row if available
            analyst_summary += "\n**Recommendation Counts:**\n"
            # Check for common recommendation fields (names might vary slightly)
            strong_buy = latest_recs.get('strongBuy', 0)
            buy = latest_recs.get('buy', 0)
            hold = latest_recs.get('hold', 0)
            sell = latest_recs.get('sell', 0)
            strong_sell = latest_recs.get('strongSell', 0)

            analyst_summary += f"- Strong Buy: {strong_buy}\n"
            analyst_summary += f"- Buy: {buy}\n"
            analyst_summary += f"- Hold: {hold}\n"
            analyst_summary += f"- Sell: {sell}\n"
            analyst_summary += f"- Strong Sell: {strong_sell}\n"
            total_analysts = strong_buy + buy + hold + sell + strong_sell
            analyst_summary += f"- Total Analysts: {total_analysts}\n"
        else:
            analyst_summary += "\n- Analyst recommendation counts not available.\n"

        # Check for target price in basic info
        target_mean = info.get('targetMeanPrice')
        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')
        current_price = info.get('currentPrice')

        if target_mean:
            analyst_summary += "\n**Price Targets:**\n"
            analyst_summary += f"- Mean Target Price: ${target_mean:.2f}\n"
            if target_high:
                analyst_summary += f"- High Target Price: ${target_high:.2f}\n"
            if target_low:
                analyst_summary += f"- Low Target Price: ${target_low:.2f}\n"
            if current_price:
                potential = ((target_mean / current_price) - 1) * 100 if current_price else 0
                analyst_summary += f"- Potential vs Mean Target: {potential:.2f}%\n"
        else:
            analyst_summary += "\n- Analyst price target data not available.\n"

    except Exception as e:
        print(f"Error fetching or processing analyst data for {topic}: {e}")
        analyst_summary = f"Could not retrieve analyst recommendation summary for {topic} due to error: {e}"

    state['messages'].append(f"Analyst Report Agent: Analyzed recommendations for {topic}.")
    # Ensure research_info list exists
    if 'research_info' not in state:
        state['research_info'] = []
    state['research_info'].append(analyst_summary)
    return state 