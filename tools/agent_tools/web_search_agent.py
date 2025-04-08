"""Agent responsible for searching the web for relevant stock information."""

from tools.web_search_tool import web_search_tool # Import the tool

def run_web_search(state):
    """Runs the web search agent node."""
    print("--- Running Web Search Agent ---")
    topic = state["topic"]
    search_results = web_search_tool.invoke(topic) # Use the tool
    state['messages'].append(f"Web Search Agent: Found information on {topic}.")
    # Ensure research_info is initialized if it's the first step
    if 'research_info' not in state or not state['research_info']:
        state['research_info'] = []
    state['research_info'].append(search_results)
    return state 