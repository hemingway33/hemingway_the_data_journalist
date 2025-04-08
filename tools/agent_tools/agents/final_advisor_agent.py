"""Agent responsible for providing the final investment recommendation."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Assume llm is configured globally or passed in
llm = Ollama(model="gemma3")

def run_final_advisor(state):
    """Runs the final advisor agent node."""
    print("--- Running Final Advisor Agent ---")
    topic = state["topic"]
    analysis = state.get("analysis", "No analysis available.")
    messages = state["messages"]
    # Extract debate points (assuming the last message from the debate agent contains the critique)
    debate_critique = "No debate critique found." 
    for msg in reversed(messages):
        if msg.startswith("Debate Agent Critique:"):
            debate_critique = msg.replace("Debate Agent Critique:\n", "")
            break

    # Refined final advice prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Investment Advisor synthesizing research, analysis, and critical debate points to provide a final recommendation.
        Consider all available information: initial research, pricing model analysis, and the debate/critique.
        Output your recommendation in the following format:
        **Recommendation:** [Buy/Sell/Hold]
        **Confidence:** [High/Medium/Low]
        **Justification:** [Brief summary of key reasons supporting the recommendation]
        **Key Risks:** [Brief summary of key risks or uncertainties]
        """
        ),
        ("human", f"""Final Investment Recommendation Request:
        Stock: {topic}
        Pricing Model Analysis: {analysis}
        Debate/Critique Summary: {debate_critique}
        Full Conversation History: {'\n'.join(messages)}

        Provide your final, structured investment recommendation.""")
    ])
    chain = prompt | llm
    final_recommendation = chain.invoke({
        "topic": topic,
        "analysis": analysis,
        "debate_critique": debate_critique,
        "messages": messages
    })

    state['messages'].append(f"Final Advisor Recommendation:\n{final_recommendation}")
    state['recommendation'] = final_recommendation # Store the full structured recommendation
    return state 