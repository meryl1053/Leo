import datetime
import logging
import os
from dotenv import dotenv_values, load_dotenv
from googlesearch import search
from groq import Groq
from json import dump, load

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Load environment variables (try multiple methods)
def load_env_vars():
    """Load environment variables using multiple fallback methods."""
    env_vars = {}
    
    # Method 1: Try dotenv_values
    try:
        env_vars = dotenv_values(".env")
        logging.info("Loaded .env using dotenv_values")
    except Exception as e:
        logging.warning(f"dotenv_values failed: {e}")
    
    # Method 2: Try load_dotenv + os.getenv (fallback)
    if not env_vars or not env_vars.get("GroqAPIKey"):
        try:
            load_dotenv()
            env_vars = {
                "Username": os.getenv("Username", "User"),
                "Assistantname": os.getenv("Assistantname", "Assistant"),
                "GroqAPIKey": os.getenv("GroqAPIKey") or os.getenv("GROQ_API_KEY")
            }
            logging.info("Loaded .env using load_dotenv")
        except Exception as e:
            logging.warning(f"load_dotenv failed: {e}")
    
    return env_vars

# Load environment variables
env_vars = load_env_vars()
Username = env_vars.get("Username", "User")
Assistantname = env_vars.get("Assistantname", "Assistant") 
GroqAPIKey = env_vars.get("GroqAPIKey")

# Validate API key
if not GroqAPIKey:
    logging.error("❌ No Groq API key found! Please check your .env file.")
    logging.error("Expected format in .env file:")
    logging.error("GroqAPIKey=your_api_key_here")
    logging.error("or")
    logging.error("GROQ_API_KEY=your_api_key_here")
    raise ValueError("Groq API key is required")

# Test API key format
if not GroqAPIKey.startswith("gsk_"):
    logging.warning("⚠️  API key doesn't start with 'gsk_' - this might be incorrect")

logging.info(f"✅ API key loaded: {GroqAPIKey[:10]}...")

# Initialize Groq client with error handling
try:
    client = Groq(api_key=GroqAPIKey)
    logging.info("✅ Groq client initialized successfully")
except Exception as e:
    logging.error(f"❌ Failed to initialize Groq client: {e}")
    raise

# System instruction for the assistant
System = (
    f"Hello, I am {Username}. You are a very accurate and advanced AI chatbot named {Assistantname} "
    f"which has real-time up-to-date information from the internet.\n"
    f"*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***\n"
    f"*** Just answer the question from the provided data in a professional way. ***"
)

# Cap the history size to prevent overflow
MAX_CHAT_HISTORY = 20

# Ensure Data directory exists
os.makedirs("Data", exist_ok=True)

# Load chat log or initialize
def load_chat_log():
    """Load chat log with proper error handling."""
    try:
        with open("Data/ChatLog.json", "r") as f:
            messages = load(f)
        logging.info(f"Loaded {len(messages)} messages from chat log")
        return messages
    except FileNotFoundError:
        logging.info("No existing chat log found, creating new one")
        messages = []
        save_chat_log(messages)
        return messages
    except Exception as e:
        logging.error(f"Error loading chat log: {e}")
        return []

def save_chat_log(messages):
    """Save chat log with error handling."""
    try:
        with open("Data/ChatLog.json", "w") as f:
            dump(messages, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving chat log: {e}")

# Google search with better error handling
def GoogleSearch(query, max_retries=3):
    """Perform Google search with retry logic."""
    for attempt in range(max_retries):
        try:
            logging.info(f"🔍 Searching for: {query} (attempt {attempt + 1})")
            results = list(search(query, advanced=True, num_results=5))
            
            if not results:
                return f"No search results found for '{query}'."
            
            answer = f"The search results for '{query}' are:\n[start]\n"
            for i, result in enumerate(results, 1):
                title = result.title or "No title"
                description = result.description or "No description"
                answer += f"🔹 Result {i}: {title}\n   📃 Description: {description}\n\n"
            answer += "[end]"
            
            logging.info(f"✅ Found {len(results)} search results")
            return answer
            
        except Exception as e:
            logging.error(f"[GoogleSearch Attempt {attempt + 1}] {e}")
            if attempt == max_retries - 1:
                return f"Unable to search for '{query}' at this time. Please try again later."
            
    return "Search service temporarily unavailable."

# Clean response
def AnswerModifier(answer):
    """Clean and format the response."""
    if not answer:
        return "I apologize, but I couldn't generate a proper response."
    
    lines = answer.split("\n")
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

# Real-time info
def Information():
    """Get current date and time information."""
    now = datetime.datetime.now()
    return (
        "Use This Real-Time Information If Needed:\n"
        f"Day: {now.strftime('%A')}\n"
        f"Date: {now.strftime('%d')}\n"
        f"Month: {now.strftime('%B')}\n"
        f"Year: {now.strftime('%Y')}\n"
        f"Time: {now.strftime('%H')} hours, {now.strftime('%M')} minutes, {now.strftime('%S')} seconds.\n"
    )

# System primer
SystemChatBot = [
    {"role": "system", "content": System},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello, how can I help you?"}
]

# Test API connection
def test_groq_connection():
    """Test if Groq API is working."""
    try:
        logging.info("🧪 Testing Groq API connection...")
        completion = client.chat.completions.create(
            model="llama3-8b-8192",  # Use smaller model for testing
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            temperature=0.1
        )
        logging.info("✅ Groq API connection successful!")
        return True
    except Exception as e:
        logging.error(f"❌ Groq API test failed: {e}")
        return False

# Main function with enhanced error handling
def RealtimeSearchEngine(prompt):
    """Enhanced RealtimeSearchEngine with better error handling."""
    global SystemChatBot
    
    if not prompt or not prompt.strip():
        return "Please provide a valid question or query."
    
    try:
        # Load current messages
        messages = load_chat_log()
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Get search results
        search_results = GoogleSearch(prompt)
        SystemChatBot.append({"role": "system", "content": search_results})
        
        # Prepare messages for API
        api_messages = (SystemChatBot + 
                       [{"role": "system", "content": Information()}] + 
                       messages[-MAX_CHAT_HISTORY:])  # Limit context
        
        logging.info("🤖 Generating response with Groq...")
        
        try:
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=api_messages,
                temperature=0.7,
                max_tokens=2048,
                stream=True,
                stop=None,
                top_p=1
            )
            
            # Collect response
            answer = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
                    
        except Exception as api_error:
            logging.error(f"[Groq API Error] {api_error}")
            
            # Try with smaller model as fallback
            try:
                logging.info("🔄 Trying with fallback model...")
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",  # Fallback to smaller model
                    messages=api_messages,
                    temperature=0.7,
                    max_tokens=1024,
                    stream=False
                )
                answer = completion.choices[0].message.content
                logging.info("✅ Fallback model succeeded")
                
            except Exception as fallback_error:
                logging.error(f"[Fallback Error] {fallback_error}")
                return "I'm sorry, I'm having trouble connecting to my AI service right now. Please try again later."
        
        # Clean and process answer
        answer = answer.strip().replace("</s>", "")
        if not answer:
            return "I couldn't generate a proper response. Please try rephrasing your question."
        
        # Save conversation
        messages.append({"role": "assistant", "content": answer})
        messages = messages[-MAX_CHAT_HISTORY:]  # Trim history
        save_chat_log(messages)
        
        # Clean up SystemChatBot
        if SystemChatBot and len(SystemChatBot) > 3:  # Keep original 3 messages
            SystemChatBot.pop()
        
        logging.info("✅ Response generated successfully")
        return AnswerModifier(answer)
        
    except Exception as e:
        logging.error(f"[RealtimeSearchEngine Error] {e}")
        return "I encountered an error while processing your request. Please try again."

# Initialization and testing
def initialize():
    """Initialize the search engine and test connections."""
    logging.info("🚀 Initializing RealtimeSearchEngine...")
    
    # Test Groq connection
    if not test_groq_connection():
        logging.error("❌ Cannot continue without working Groq API")
        return False
    
    # Test search functionality
    try:
        test_search = GoogleSearch("test", max_retries=1)
        if "Unable to search" not in test_search:
            logging.info("✅ Google search is working")
        else:
            logging.warning("⚠️  Google search may have issues")
    except Exception as e:
        logging.warning(f"⚠️  Search test failed: {e}")
    
    logging.info("✅ RealtimeSearchEngine initialized successfully")
    return True

# For testing
if __name__ == "__main__":
    if initialize():
        print("RealtimeSearchEngine is ready!")
        while True:
            try:
                query = input("\nEnter your query (or 'quit' to exit): ")
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query.strip():
                    response = RealtimeSearchEngine(query)
                    print(f"\n{response}")
                else:
                    print("Please enter a valid query.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        print("❌ Failed to initialize RealtimeSearchEngine")
