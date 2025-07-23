from groq import Groq
from json import dump, load
import datetime
from dotenv import dotenv_values  # Importing dotenv to load environment variables

# Load environment variables from .env file
env_vars = dotenv_values(".env")

# Retrive specific environment variables from username, assisstant name, and API key
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

# initialize Groq client with the API key
client = Groq(api_key=GroqAPIKey)

# Initialize an empty list to store chat messages
messages = []

# Define a system message that provides context to the AI chatbot about its role and behavior
System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** Reply in only English, even if the question is in Hindi, reply in English.***
*** Do not provide notes in the output, just answer the question and never mention your training data. ***
"""

# A list of system instruction for the chatbot
SystemChatBot = [
    {"role": "system", "content": System}
]

# Attempt to load the chat log from a JSON file
try:
    with open(r"Data/ChatLog.json", "r") as f:
        messages = load(f)
except FileNotFoundError:
    # If the file does not exist, create an empty JSON file to store chat logs
    with open(r"Data/how lod is aniketChatLog.json", "w") as f:
        dump([], f)
        
# Function to get real-time date and time information
def RealtimeInformation():
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")
    
    # Format the information into a string
    data = f"Please use this real-time information if needed, \n"
    data += f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
    data += f"Time: {hour} hours :{minute} minutes :{second} seconds.\n"
    return data

# Function to modify the chatbot's response for better formatting
def AnswerModifier(Answer):
    lines = Answer.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = "\n".join(non_empty_lines)
    return modified_answer

# Main chatbot function to handle user queries
def ChatBot(Query):
    """ This function sends the user's query to the chatbot and returns the AI's response."""
    
    try:
        # load the existing chat log from JSON file
        with open(r"Data/ChatLog.json", "r") as f:
            messages = load(f)
            
        # Append the user's query to the message list
        messages.append({"role": "user", "content": f"{Query}"})
        
        # Make a request to the Groq API for a response
        completion = client.chat.completions.create(
            model = "llama3-70b-8192",
            messages = SystemChatBot + [{"role": "system", "content": RealtimeInformation()}] + messages,
            temperature = 0.7,
            max_tokens = 1024,
            stream = True,
            stop = None,
            top_p=1
        )
        
        Answer = ""
        
        # Process the streamed response chunks
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content
                
        Answer = Answer.replace("</s>", "")
        
        # Append the chatbot's response to the message list
        messages.append({"role": "assistant", "content": Answer})
        
        # Save the updated chat log back to the JSON file
        with open(r"Data/ChatLog.json", "w") as f:
            dump(messages, f, indent=4)
            
        # Return the formatted response
        return AnswerModifier(Answer=Answer)
    
    except Exception as e:
        print(f"[ERROR] {e}")
        
        # Reset the chat log if error occurred (likely corrupted or empty file)
        try:
            with open(r"Data/ChatLog.json", "w") as f:
                dump([], f, indent=4)
            print("[INFO] Chat log reset successfully.")
        except Exception as file_error:
            print(f"[ERROR] Failed to reset chat log: {file_error}")
        
        return "[ERROR] Chatbot failed to process your request. Please try again."

        
# Main program entry point

if __name__ == "__main__":
    while True:
        user_input = input("Enter your Question: ")
        print(ChatBot(user_input))
