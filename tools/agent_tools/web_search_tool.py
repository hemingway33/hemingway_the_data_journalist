import os
from tavily import TavilyClient
from langchain_core.tools import tool

# Ensure you have TAVILY_API_KEY set in your environment variables
os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY" # Replace with your actual key or load from env

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

@tool("web_search", return_direct=False)
def web_search_tool(query: str):
    """Performs a web search using Tavily for the given query."""
    try:
        # Example: Search for relevant company news and financial summaries
        results = tavily_client.search(query=f"{query} stock news and financial overview", search_depth="advanced")
        # Process results - extracting key information
        content = [f"URL: {res.get('url', 'N/A')}\nContent: {res.get('content', '')[:500]}..." for res in results.get('results', [])]
        return "\n\n".join(content) if content else "No results found."
    except Exception as e:
        return f"Error during Tavily search: {e}"

# Example usage:
if __name__ == "__main__":
    search_results = web_search_tool.invoke("Apple Inc.")
    print(search_results) 