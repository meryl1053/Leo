import cohere
from dotenv import dotenv_values
import traceback
# Load environment variables
env_vars = dotenv_values(".env")
CohereAPIKey = env_vars.get("CohereAPIKey")

# Initialize Cohere client
co = cohere.Client(api_key=CohereAPIKey)

# Task categories to recognize
TASK_KEYWORDS = [
    "exit", "general", "realtime", "open", "close", "play", 
    "generate image", "system", "content", "google search", 
    "youtube search", "reminder", "data analysis", "agent", "3d model"
]

# Core rules for fast classification
def rule_based_filter(query: str) -> str:
    q = query.lower()

    if "3d model" in q or "generate a model" in q or "create 3d object" in q:
        return "content 3d model"
    if "create agent" in q or "build agent" in q:
        return "agent"
    if "analyze" in q or "csv" in q or "dataset" in q:
        return "data analysis"
    if "generate image" in q or "draw" in q or "create image" in q:
        return "generate image"
    return ""  # No rule matched, pass to LLM

# LLM-based decision using Cohere
def llm_decision(query: str) -> str:
    preamble = """
You are a decision-making engine. Given a user query, classify it into one or more tasks:
- 'general'
- 'realtime'
- 'open (app)'
- 'close (app)'
- 'play (song)'
- 'generate image (prompt)'
- 'reminder (datetime + message)'
- 'system (action)'
- 'content (topic)'
- 'google search (topic)'
- 'youtube search (topic)'
- 'exit'
Respond with one or more task labels, comma-separated if needed.
Never answer the query, only classify it.
"""

    response = co.chat(
        model="command-r-plus",
        message=query,
        preamble=preamble,
        temperature=0.3
    )

    return response.text.strip()

# Main function for deciding intent
def FirstLayerDMM(query: str) -> list:
    query = query.strip().lower()

    # 1. Fast rule-based filtering
    rule_result = rule_based_filter(query)
    if rule_result:
        return [rule_result]

    # 2. Fallback to LLM classification
    llm_result = llm_decision(query)
    return [llm_result]

# CLI test mode
if __name__ == "__main__":
    try:
        while True:
            user_input = input(">>> ")
            result = FirstLayerDMM(user_input)
            print("[Brain 🧠] Decision:", result)
    except Exception as e:
        traceback.print_exc()
