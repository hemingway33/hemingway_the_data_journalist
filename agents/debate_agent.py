"""Agent responsible for debating the investment thesis."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Assume llm is configured globally or passed in
llm = Ollama(model="gemma3")

def run_debate(state):
    """Runs the debate agent node."""
    print("--- Running Debate Agent ---")
    topic = state["topic"]
    analysis = state.get("analysis", "No prior analysis provided.") # Use .get for safety
    messages = state["messages"]

    # Refined debate prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a meticulous and critical financial analyst reviewing an investment thesis.
        Your goal is to rigorously challenge the analysis provided. Identify potential flaws, overlooked risks, weak assumptions, data gaps, or alternative interpretations.
        Consider the research gathered and the pricing model analysis.
        Structure your critique clearly."""),
        ("human", f"""Investment Thesis Review:
        Stock: {topic}
        Provided Analysis: {analysis}
        Supporting Research Summary (from previous steps): {'\n'.join(state.get('research_info', []))}
        Conversation History: {'\n'.join(messages)}

        Critique the investment thesis. What are the strongest counterarguments or risks? Are there critical missing pieces of information?""")
    ])
    chain = prompt | llm
    debate_notes = chain.invoke({
        "topic": topic,
        "analysis": analysis,
        "research_info": state.get('research_info', []),
        "messages": messages
    })

    state['messages'].append(f"Debate Agent Critique:\n{debate_notes}")
    # Optionally, the debate notes could modify the 'analysis' state, but for now, just append.
    return state 