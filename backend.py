# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step1: Setup Pydantic Model (Schema Validation)
from pydantic import BaseModel
from typing import List, Dict, Any

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


# Step2: Setup AI Agent from FrontEnd Request
from fastapi import FastAPI, HTTPException
from ai_agent import get_response_from_ai_agent


ALLOWED_MODEL_NAMES = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
    "gemini-1.5-pro",
]  # Add Gemini model


app = FastAPI(title="LangGraph AI Agent")


@app.post("/chat")
def chat_endpoint(request: RequestState) -> Dict[str, Any]:
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail="Invalid model name. Kindly select a valid AI model",
        )

    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    try:
        # Create AI Agent and get response from it!
        response = get_response_from_ai_agent(
            llm_id, query, allow_search, system_prompt, provider
        )
        return {"response": response}  # Return the response in a dictionary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Step3: Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)