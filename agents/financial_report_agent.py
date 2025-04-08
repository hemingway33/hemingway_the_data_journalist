"""Agent responsible for reading and summarizing financial reports (e.g., 10-K, 10-Q)."""

import yfinance as yf
import pandas as pd

def format_value(value):
    """Helper function to format large numbers."""
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.2f}K"
    return f"${value:.2f}"

def run_financial_report_analysis(state):
    """Runs the financial report analysis agent node using yfinance data."""
    print("--- Running Financial Report Agent ---")
    topic = state["topic"]

    try:
        stock = yf.Ticker(topic)

        # Get financial statements (usually annually)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        report_summary = f"Financial Report Summary for {topic} (Based on latest yfinance data):\n"

        if not financials.empty:
            latest_financials = financials.iloc[:, 0] # Get the most recent column
            report_summary += "\n**Income Statement Highlights:**\n"
            report_summary += f"- Total Revenue: {format_value(latest_financials.get('Total Revenue', 'N/A'))}\n"
            report_summary += f"- Gross Profit: {format_value(latest_financials.get('Gross Profit', 'N/A'))}\n"
            report_summary += f"- Operating Income: {format_value(latest_financials.get('Operating Income', 'N/A'))}\n"
            report_summary += f"- Net Income: {format_value(latest_financials.get('Net Income', 'N/A'))}\n"
        else:
            report_summary += "\n- Income Statement data not available.\n"

        if not balance_sheet.empty:
            latest_balance_sheet = balance_sheet.iloc[:, 0]
            report_summary += "\n**Balance Sheet Highlights:**\n"
            report_summary += f"- Total Assets: {format_value(latest_balance_sheet.get('Total Assets', 'N/A'))}\n"
            report_summary += f"- Total Liabilities Net Minority Interest: {format_value(latest_balance_sheet.get('Total Liabilities Net Minority Interest', 'N/A'))}\n"
            report_summary += f"- Total Equity Gross Minority Interest: {format_value(latest_balance_sheet.get('Total Equity Gross Minority Interest', 'N/A'))}\n"
            # Note: yfinance might use slightly different names for fields
            # report_summary += f"- Total Debt: {format_value(latest_balance_sheet.get('Total Debt', 'N/A'))}\n" # Often needs calculation
            report_summary += f"- Cash and Cash Equivalents: {format_value(latest_balance_sheet.get('Cash And Cash Equivalents', 'N/A'))}\n"
        else:
            report_summary += "\n- Balance Sheet data not available.\n"

        if not cash_flow.empty:
            latest_cash_flow = cash_flow.iloc[:, 0]
            report_summary += "\n**Cash Flow Highlights:**\n"
            report_summary += f"- Cash Flow from Operations: {format_value(latest_cash_flow.get('Operating Cash Flow', 'N/A'))}\n"
            report_summary += f"- Cash Flow from Investing: {format_value(latest_cash_flow.get('Investing Cash Flow', 'N/A'))}\n"
            report_summary += f"- Cash Flow from Financing: {format_value(latest_cash_flow.get('Financing Cash Flow', 'N/A'))}\n"
            report_summary += f"- Free Cash Flow: {format_value(latest_cash_flow.get('Free Cash Flow', 'N/A'))}\n"
        else:
            report_summary += "\n- Cash Flow data not available.\n"

    except Exception as e:
        print(f"Error fetching or processing financial data for {topic}: {e}")
        report_summary = f"Could not retrieve financial report summary for {topic} due to error: {e}"

    state['messages'].append(f"Financial Report Agent: Analyzed reports for {topic}.")
    # Ensure research_info list exists
    if 'research_info' not in state:
        state['research_info'] = []
    state['research_info'].append(report_summary)
    return state 