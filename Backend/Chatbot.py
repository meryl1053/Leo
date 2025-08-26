from groq import Groq
from json import dump, load
import datetime
import os
from dotenv import dotenv_values
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneralChatBot:
    def __init__(self):
        """Initialize the chatbot with configuration and setup."""
        try:
            self.load_config()
            self.setup_client()
            self.setup_data_directory()
            self.chat_log_path = "Data/ChatLog.json"
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            self._initialized = False
            # Set fallback values for testing
            self.username = "User"
            self.assistant_name = "Assistant"
            self.groq_api_key = None
            self.client = None
            self.chat_log_path = "Data/ChatLog.json"
        
    def load_config(self):
        """Load configuration from environment variables."""
        try:
            env_vars = dotenv_values(".env")
            self.username = env_vars.get("Username", "User")
            self.assistant_name = env_vars.get("Assistantname", "Assistant")
            self.groq_api_key = env_vars.get("GroqAPIKey")
            
            if not self.groq_api_key:
                raise ValueError("GroqAPIKey not found in environment variables")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def setup_client(self):
        """Initialize the Groq client."""
        try:
            self.client = Groq(api_key=self.groq_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def setup_data_directory(self):
        """Create data directory if it doesn't exist."""
        if not os.path.exists("Data"):
            os.makedirs("Data")
            logger.info("Created Data directory")
    
    def get_system_prompt(self):
        """Generate the system prompt for the chatbot."""
        return f"""You are {self.assistant_name}, an intelligent and helpful AI assistant created to help {self.username}.

Key guidelines:
- Provide accurate, informative, and concise responses
- Be helpful and friendly while maintaining professionalism  
- Answer questions directly without unnecessary elaboration
- Use clear and simple language
- If you don't know something, admit it rather than guessing
- Stay focused on the user's question
- Respond in English regardless of the input language
- Avoid mentioning your training data or technical limitations unless directly asked

You excel at:
- General knowledge questions
- Explanations and clarifications
- Problem-solving assistance
- Creative tasks
- Analysis and reasoning
- Educational support

Remember: Focus on being genuinely helpful rather than just providing information."""

    def load_chat_history(self):
        """Load existing chat history from file."""
        try:
            if os.path.exists(self.chat_log_path):
                with open(self.chat_log_path, "r", encoding="utf-8") as f:
                    return load(f)
            else:
                return []
        except Exception as e:
            logger.warning(f"Failed to load chat history: {e}")
            return []
    
    def save_chat_history(self, messages):
        """Save chat history to file."""
        try:
            with open(self.chat_log_path, "w", encoding="utf-8") as f:
                dump(messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
    
    def clean_response(self, response):
        """Clean and format the response."""
        # Remove unwanted tokens and clean up formatting
        response = response.replace("</s>", "").strip()
        
        # Remove excessive newlines
        lines = [line.strip() for line in response.split("\n")]
        cleaned_lines = []
        
        for line in lines:
            if line or (cleaned_lines and cleaned_lines[-1]):  # Keep non-empty lines and single empty lines
                cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines).strip()
    
    def manage_conversation_length(self, messages, max_messages=20):
        """Keep conversation history manageable by limiting message count."""
        if len(messages) > max_messages:
            # Keep first few messages for context and recent messages
            return messages[:2] + messages[-(max_messages-2):]
        return messages
    
    def get_response(self, query):
        """Get response from the chatbot for a given query."""
        try:
            # Load existing conversation
            messages = self.load_chat_history()
            
            # Add user message
            messages.append({"role": "user", "content": query})
            
            # Manage conversation length
            messages = self.manage_conversation_length(messages)
            
            # Prepare system message
            system_messages = [{"role": "system", "content": self.get_system_prompt()}]
            
            # Make API request
            completion = self.client.chat.completions.create(
                model="llama3-70b-8192",  # You can change this to other available models
                messages=system_messages + messages,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
                stop=None,
                top_p=0.9
            )
            
            # Collect streamed response
            response = ""
            try:
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                return "Sorry, I encountered an error while processing your request. Please try again."
            
            # Clean the response
            response = self.clean_response(response)
            
            if not response:
                return "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
            
            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response})
            
            # Save updated conversation
            self.save_chat_history(messages)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."
    
    def clear_history(self):
        """Clear the conversation history."""
        try:
            with open(self.chat_log_path, "w", encoding="utf-8") as f:
                dump([], f)
            logger.info("Chat history cleared successfully")
            return "Chat history cleared successfully!"
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return "Failed to clear chat history."
    
    def get_conversation_stats(self):
        """Get statistics about the current conversation."""
        messages = self.load_chat_history()
        user_messages = len([msg for msg in messages if msg["role"] == "user"])
        assistant_messages = len([msg for msg in messages if msg["role"] == "assistant"])
        
        return f"Conversation Stats:\n- User messages: {user_messages}\n- Assistant messages: {assistant_messages}\n- Total messages: {len(messages)}"

# Compatibility alias for tests
class ChatBot(GeneralChatBot):
    """Compatibility wrapper for GeneralChatBot"""
    
    def __init__(self):
        super().__init__()
    
    def is_ready(self):
        """Check if chatbot is ready to use"""
        return getattr(self, '_initialized', False) and self.client is not None


def main():
    """Main function to run the chatbot."""
    try:
        chatbot = GeneralChatBot()
        print(f"\n🤖 {chatbot.assistant_name} is ready to help!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'stats' to see conversation statistics.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input(f"\n{chatbot.username}: ").strip()
                
                if not user_input:
                    print("Please enter a question or message.")
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(f"\n{chatbot.assistant_name}: Goodbye! Have a great day! 👋")
                    break
                elif user_input.lower() == 'clear':
                    print(f"\n{chatbot.assistant_name}: {chatbot.clear_history()}")
                    continue
                elif user_input.lower() == 'stats':
                    print(f"\n{chatbot.assistant_name}: {chatbot.get_conversation_stats()}")
                    continue
                
                # Get and display response
                response = chatbot.get_response(user_input)
                print(f"\n{chatbot.assistant_name}: {response}")
                
            except KeyboardInterrupt:
                print(f"\n\n{chatbot.assistant_name}: Goodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"An error occurred: {e}")
                
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        print("Please check your .env file and ensure all required variables are set.")

if __name__ == "__main__":
    main()
