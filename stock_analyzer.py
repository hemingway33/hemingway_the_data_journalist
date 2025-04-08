"""
Main application file for the multi-agent stock analyzer.
"""

import os
import operator
from typing import TypedDict, Annotated, List

from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END

# Import agent functions
from agents.web_search_agent import run_web_search
from agents.financial_report_agent import run_financial_report_analysis
from agents.analyst_report_agent import run_analyst_report_analysis
from agents.pricing_model_agent import run_pricing_model
from agents.debate_agent import run_debate
from agents.final_advisor_agent import run_final_advisor

# TODO: Import tools for other agents (financial data, report parsing)

# Configure the LLM (used by debate and final advisor agents for now)
# Ensure Ollama is running and the model (e.g., gemma3) is pulled
llm = Ollama(model="gemma3")

# Define the state for the graph
class AgentState(TypedDict):
    topic: str
    research_info: List[str] = [] # Initialize as empty list
    analysis: str = ""
    recommendation: str = ""
    messages: Annotated[List[str], operator.add]

# Define Agent Nodes are the imported functions

# Define the graph workflow
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("web_search", run_web_search)
workflow.add_node("financial_reports", run_financial_report_analysis)
workflow.add_node("analyst_reports", run_analyst_report_analysis)
workflow.add_node("pricing_model", run_pricing_model)
workflow.add_node("debate", run_debate)
workflow.add_node("final_advice", run_final_advisor)

# Define edges (sequential flow for now)
workflow.set_entry_point("web_search")
workflow.add_edge("web_search", "financial_reports")
workflow.add_edge("financial_reports", "analyst_reports")
workflow.add_edge("analyst_reports", "pricing_model")
workflow.add_edge("pricing_model", "debate")
workflow.add_edge("debate", "final_advice")
workflow.add_edge("final_advice", END)

# Compile the graph
app = workflow.compile()

# Run the analysis (example)
if __name__ == "__main__":
    stock_topic = "NVIDIA" # Example topic
    print(f"--- Starting Analysis for {stock_topic} ---")
    initial_state = {
        "topic": stock_topic,
        "messages": [f"Starting analysis for {stock_topic}"],
        # Ensure keys exist even if empty, matching AgentState
        "research_info": [],
        "analysis": "",
        "recommendation": ""
    }
    # The invoke method streams intermediate results by default
    # Use stream or batch for different execution patterns if needed
    final_result = app.invoke(initial_state)

    print("\n--- Analysis Complete ---")
    print(f"Final Recommendation for {stock_topic}:")
    print(final_result.get('recommendation', 'No recommendation generated.'))
    print("\n--- Message History ---")
    for msg in final_result.get('messages', []):
        print(msg)
    # print("\n--- Full Final State ---")
    # print(final_result) 