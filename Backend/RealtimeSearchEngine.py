from googlesearch import search
from groq import Groq
from json import dump, load
import datetime
from dotenv import dotenv_values  # Importing dotenv to load environment variables

# Load environment variables from .env file
env_vars = dotenv_values(".env")

# Retrieve environment variables for the chatbot configuration
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

# Initialize Groq client with the API key
client = Groq(api_key=GroqAPIKey)

# Define the system instruction for the chatbot
System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which has real-time up-to-date information from the internet.
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***
*** Just answer the question from the provided data in a professional way. ***"""

# Try to load the chat log from a JSON file, or creatre an empty one if it doesn't exist
try:
    with open(r"Data/ChatLog.json", "r") as f:
        messages = load(f)
except:
    with open(r"Data/ChatLog.json", "w") as f:
        dump([], f)
        
# Function to perform a Google search and format the top results
def GoogleSearch(query):
    results = list(search(query, advanced=True, num_results=5))
    Answer = f"The search results for '{query}' are:\n[start]\n"
    
    for i in results:
        Answer += f"Title: {i.title}\nDescription: {i.description}\n\n"
        
    Answer += "[end]"
    return Answer

# Function to clean up the answer by removing empty lines
def AnswerModifier(Answer):
    lines = Answer.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = "\n".join(non_empty_lines)
    return modified_answer


# Predefined chatbot conversation system message and an initial user message
SystemChatBot = [
    {"role": "system", "content": System},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello, how can i help you?"}
]


# Function to get real-time date and time information
def Information():
    data = ""
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")
    data += f"Use This Real-Time Information If Needed, \n"
    data += f"Day: {day}\n"
    data += f"Date: {date}\n"
    data += f"Month: {month}\n"
    data += f"Year: {year}\n"
    data += f"Time: {hour} hours, {minute} minutes, {second} seconds.\n"
    return data

# Function to handle real-time search and response generation
def RealtimeSearchEngine(prompt):
    global messages,SystemChatBot
    
    # Load the chat log from the JSON file
    with open(r"Data/ChatLog.json", "r") as f:
        messages = load(f)
        
    # Append the user's query to the message list
    messages.append({"role": "user", "content": f"{prompt}"})
    
    # Add Google search results to the system chatbot messages
    SystemChatBot.append({"role": "system", "content": GoogleSearch(prompt)})
    
    # Generate a response using the Groq API
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=SystemChatBot + [{"role": "system", "content": Information()}] + messages,
        temperature=0.7,
        max_tokens=2048,
        stream=True,
        stop=None,
        top_p=1
    )
    
    Answer = ""
    
    # Concatenate response chunks from streaming output
    for chunk in completion:
        if chunk.choices[0].delta.content:
            Answer += chunk.choices[0].delta.content
            
    # Clean up the response
    Answer = Answer.strip().replace("</s>", "")
    messages.append({"role": "assistant", "content": Answer})
    
    # Save the updated chat log back to the JSON file
    with open(r"Data/ChatLog.json", "w") as f:
        dump(messages, f, indent=4)
        
    # Remove the most recent system message from the chatbot conversation
    SystemChatBot.pop()
    return AnswerModifier(Answer=Answer)  # Return the cleaned-up answer

# Main entry point of the program for interactive querying
if __name__ == "__main__":
    while True:
        prompt = input("Enter your query: ")
        print(RealtimeSearchEngine(prompt))
