import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini 2.5 Flash — shared reasoning model for all Agents
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )
