# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()


# Step1: Setup UI with streamlit (model provider, model, system prompt, web_search, query)
import streamlit as st
import requests
import json  # Import json module

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

system_prompt = st.text_area(
    "Define your AI Agent: ",
    height=70,
    placeholder="Type your system prompt here...",
)

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_GEMINI = ["gemini-1.5-pro"]  # Gemini model

provider = st.radio("Select Provider:", ("Groq", "Gemini"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "Gemini":
    selected_model = st.selectbox("Select Gemini Model:", MODEL_NAMES_GEMINI)


allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area(
    "Enter your query: ", height=150, placeholder="Ask Anything!"
)

API_URL = "http://127.0.0.1:8000/chat"

if st.button("Ask Agent!"):
    if user_query.strip():
        # Step2: Connect with backend via URL

        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search,
        }

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()

            if "error" in response_data:
                st.error(response_data["error"])
            elif "response" in response_data:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response_data['response']}")
            else:
                st.error("Unexpected response format from the backend.")

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred during the request: {e}")
        except json.JSONDecodeError:
            st.error("Failed to decode JSON response from the backend.")