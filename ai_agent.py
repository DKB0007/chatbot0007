# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step1: Setup API Keys for Groq and Tavily
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GEIMINI_API_KEY = os.environ.get("GEIMINI_API_KEY")


# Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize Google Gemini LLM


# Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from typing import List, Dict, Any


system_prompt = "Act as an AI chatbot who is smart and friendly"


def get_response_from_ai_agent(
    llm_id: str, query: List[str], allow_search: bool, system_prompt: str, provider: str
) -> str:
    """
    Retrieves a response from an AI agent.

    Args:
        llm_id: The ID of the language model to use.
        query: The user's query as a list of strings.
        allow_search: Whether to allow the agent to use web search.
        system_prompt: The system prompt to guide the agent.
        provider: The provider of the language model (e.g., "Groq", "Gemini").

    Returns:
        The AI agent's response as a string.
    """

    if provider == "Groq":
        llm = ChatGroq(model=llm_id, groq_api_key=GROQ_API_KEY)
    elif provider == "Gemini":
        llm = ChatGoogleGenerativeAI(model=llm_id, google_api_key=GEIMINI_API_KEY)
    else:
        raise ValueError(f"Invalid provider: {provider}.  Must be 'Groq' or 'Gemini'.")

    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    agent = create_react_agent(model=llm, tools=tools, state_modifier=system_prompt)
    state: Dict[str, Any] = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    if messages:
        ai_messages = [
            message.content for message in messages if isinstance(message, AIMessage)
        ]
        return ai_messages[-1]
    else:
        return "No response from the agent."