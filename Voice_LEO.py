#!/usr/bin/env python3
"""
🎤 Voice-Only LEO AI Assistant 🎤
Streamlined version focused on voice interaction with GUI integration
"""

import asyncio
import logging
import os
import sys
import json
import threading
import time
import subprocess
import platform
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
import traceback
import shlex

# Add paths
sys.path.append(str(Path(__file__).parent / "Backend"))
sys.path.append(str(Path(__file__).parent / "Frontend"))

# GUI Integration
from Frontend.GUI import GraphicalUserInterface, Speak, SPEECH_RECOGNITION_AVAILABLE, TTS_AVAILABLE

# Enhanced Speech Recognition
from Frontend.EnhancedSpeechRecognition import EnhancedListen, get_speech_stats

# Core Backend Components (enhanced with full capabilities)
try:
    from Backend.RealtimeSearchEngine import RealtimeSearchEngine, GoogleSearch, test_groq_connection
    SEARCH_AVAILABLE = True
except ImportError:
    print("⚠️ Search engine not available")
    SEARCH_AVAILABLE = False

try:
    from Backend.Chatbot import ChatBot
    CHATBOT_AVAILABLE = True
except ImportError:
    print("⚠️ Chatbot not available")
    CHATBOT_AVAILABLE = False

# Enhanced capabilities
try:
    from Backend.ImageGeneration import EnhancedTextToImage
    IMAGE_GENERATION_AVAILABLE = True
except ImportError:
    print("⚠️ Image generation not available")
    IMAGE_GENERATION_AVAILABLE = False

try:
    from Backend.Automation import Content, PlayMusic, YouTubeSearch, PlayYoutube
    AUTOMATION_AVAILABLE = True
except ImportError:
    print("⚠️ Automation features not available")
    AUTOMATION_AVAILABLE = False

try:
    from Backend.TextToSpeech import TextToSpeech as TTSEngine
    ENHANCED_TTS_AVAILABLE = True
except ImportError:
    print("⚠️ Enhanced TTS not available")
    ENHANCED_TTS_AVAILABLE = False

# Advanced Analytics
try:
    from Backend.DataAnalyzer import UniversalAutonomousResearchAnalytics
    DATA_ANALYZER_AVAILABLE = True
except ImportError:
    print("⚠️ Data Analyzer not available")
    DATA_ANALYZER_AVAILABLE = False

# 3D Model Generation
try:
    from Backend.ModelMaker import PromptEnhancer, GenerationConfig, PreviewLauncher
    import Backend.ModelMaker as ModelMaker
    MODEL_MAKER_AVAILABLE = True
except ImportError:
    print("⚠️ 3D Model Maker not available")
    MODEL_MAKER_AVAILABLE = False

# Multi-Agent System
try:
    from Backend.UltraAdvancedAgentCreator import EnhancedAgentSystem, SystemConfig, AgentRole, TaskPriority, create_enhanced_config
    AGENT_SYSTEM_AVAILABLE = True
except ImportError:
    print("⚠️ Agent Creator not available")
    AGENT_SYSTEM_AVAILABLE = False

# Master Integration System
try:
    from Backend.MasterIntegrationSystem import MasterIntegrationSystem, SystemMode, IntegrationStatus
    MASTER_INTEGRATION_AVAILABLE = True
except ImportError:
    print("⚠️ Master Integration System not available")
    MASTER_INTEGRATION_AVAILABLE = False

# Advanced Voice Triggers
try:
    from Backend.SoundTrigger import VoiceActivationSystem, Config as VoiceConfig
    SOUND_TRIGGER_AVAILABLE = True
except ImportError:
    print("⚠️ Sound Trigger not available")
    SOUND_TRIGGER_AVAILABLE = False

# Auto-Updater
try:
    from Backend.AutoUpdater import IntelligentAIUpdater
    AUTO_UPDATER_AVAILABLE = True
except ImportError:
    print("⚠️ Auto Updater not available")
    AUTO_UPDATER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_leo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VoiceLEO:
    """Enhanced Voice LEO Assistant with Master Integration System"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.chatbot = None
        self.search_engine = None
        self.interaction_count = 0
        self.start_time = datetime.now()
        
        # Master Integration System
        self.master_system = None
        
        # Agent System
        self.agent_system = None
        
        # Voice-focused configuration
        self.config = {
            "voice_enabled": True,
            "continuous_listening": True,
            "auto_response": True,
            "search_integration": True,
            "gui_mode": True,
            "always_listening": True,
            "master_integration": True,
            "learning_enabled": True,
            "resource_optimization": True,
            "multimodal_processing": True
        }
        
        # Statistics
        self.stats = {
            "voice_interactions": 0,
            "successful_responses": 0,
            "average_response_time": 0.0,
            "master_system_requests": 0,
            "learning_improvements": 0
        }
    
    async def initialize(self):
        """Initialize enhanced voice system with Master Integration"""
        try:
            self.logger.info("🎤 Initializing Enhanced Voice LEO...")
            
            # Check voice capabilities
            if not SPEECH_RECOGNITION_AVAILABLE:
                self.logger.error("❌ Speech recognition not available!")
                return False
            
            # Initialize Master Integration System if available
            if MASTER_INTEGRATION_AVAILABLE and self.config["master_integration"]:
                self.logger.info("🌟 Initializing Master Integration System...")
                self.master_system = MasterIntegrationSystem()
                await self.master_system.initialize()
                
                # Set system mode to voice assistant
                self.master_system.current_mode = SystemMode.VOICE_ASSISTANT
                self.logger.info("✅ Master Integration System initialized for voice mode")
            else:
                self.logger.info("⚠️ Master Integration System not available, using basic components")
            
            # Initialize Agent System
            if AGENT_SYSTEM_AVAILABLE:
                try:
                    config_path = Path("config.yaml")
                    if not config_path.exists():
                        self.logger.info(f"'{config_path}' not found, creating one.")
                        create_enhanced_config()

                    self.logger.info("🌟 Initializing Enhanced Agent System...")
                    config = SystemConfig.load_from_file(str(config_path))
                    self.agent_system = EnhancedAgentSystem(config)
                    await self.agent_system.initialize()
                    self.logger.info("✅ Enhanced Agent System initialized")
                except Exception as e:
                    self.logger.error(f"❌ Agent System initialization failed: {e}", exc_info=True)
                    self.agent_system = None # ensure it's None on failure
            
            # Initialize search engine
            if SEARCH_AVAILABLE:
                if test_groq_connection():
                    self.search_engine = "available"  # Just mark as available
                    self.logger.info("✅ Search engine initialized")
            
            # Initialize chatbot
            if CHATBOT_AVAILABLE:
                self.chatbot = ChatBot()
                self.logger.info("✅ Chatbot initialized")
            
            self.is_running = True
            self.logger.info("✅ Enhanced Voice LEO ready with all capabilities!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            traceback.print_exc()
            return False
    
    async def process_voice_command(self, command: str) -> str:
        """Process voice command and return response with enhanced capabilities"""
        try:
            self.interaction_count += 1
            self.stats["voice_interactions"] += 1
            
            # Try Master Integration System first if available
            if self.master_system and self.master_system.status == IntegrationStatus.CONNECTED:
                try:
                    self.stats["master_system_requests"] += 1
                    
                    # Process through Master Integration System
                    response = await self.master_system.process_unified_request(
                        user_input=command,
                        user_id="voice_user",
                        context={"input_type": "voice", "continuous_listening": True}
                    )
                    
                    if response and response.get("success"):
                        self.stats["successful_responses"] += 1
                        return response.get("response", "I processed your request but couldn't generate a response.")
                    else:
                        # Fall back to basic processing if Master System fails
                        self.logger.warning("Master System failed, falling back to basic processing")
                        
                except Exception as e:
                    self.logger.error(f"Master Integration System error: {e}")
                    # Continue to fallback processing
            
            # Fallback to basic command routing
            command_lower = command.lower().strip()
            
            # Basic commands
            if any(word in command_lower for word in ['hello', 'hi', 'hey']):
                return "Hello! I'm LEO, your enhanced voice assistant. I can search, chat, create images, play music, write content, and much more!"
            
            elif any(word in command_lower for word in ['time', 'what time']):
                current_time = datetime.now().strftime("%I:%M %p")
                return f"The current time is {current_time}"
            
            elif any(word in command_lower for word in ['date', 'what date', 'today']):
                current_date = datetime.now().strftime("%A, %B %d, %Y")
                return f"Today is {current_date}"
            
            # Enhanced capabilities
            elif any(word in command_lower for word in ['create image', 'generate image', 'make image', 'draw']):
                return await self.handle_image_generation(command)
            
            elif any(word in command_lower for word in ['play music', 'play song', 'music']):
                return await self.handle_music(command)
            
            elif any(word in command_lower for word in ['youtube', 'play video']):
                return await self.handle_youtube(command)
            
            elif any(word in command_lower for word in ['write', 'content', 'create content', 'letter', 'essay', 'poem']):
                return await self.handle_content_creation(command)
            
            # Advanced Analytics
            elif any(word in command_lower for word in ['analyze data', 'analyze my data', 'run analytics', 'data analysis', 'analyze file']):
                return await self.handle_data_analysis(command)
            
            # 3D Model Generation
            elif any(word in command_lower for word in ['create 3d model', 'generate 3d', 'make 3d model', '3d model']):
                return await self.handle_3d_model_generation(command)
            
            # Multi-Agent System
            elif 'task status' in command_lower:
                return await self.handle_get_task_status(command)
            elif any(word in command_lower for word in ['create agent', 'deploy agent', 'agent team', 'create ai agent']):
                return await self.handle_agent_task(command)
            
            # System Updates and Management
            elif any(word in command_lower for word in ['update yourself', 'learn new features', 'backup system', 'system update']):
                return await self.handle_system_updates(command)
            
            # APPLICATION CONTROL - Open/Launch Apps
            elif any(word in command_lower for word in ['open', 'launch', 'start', 'run']) and not any(word in command_lower for word in ['downloads', 'documents', 'desktop', 'home', 'applications']):
                return await self.handle_app_control(command)
            
            # SYSTEM CONTROL - Volume, Brightness, WiFi, etc.
            elif any(word in command_lower for word in ['volume', 'brightness', 'wifi', 'bluetooth', 'sleep', 'lock', 'dock', 'mission control', 'desktop', 'trash', 'mute', 'unmute', 'brighter', 'dimmer', 'louder', 'quieter']):
                return await self.handle_system_control(command)
            
            # FILE OPERATIONS - Folders and Files
            elif any(word in command_lower for word in ['open downloads', 'open documents', 'open desktop', 'open home', 'open applications', 'create folder', 'new folder', 'make folder']):
                return await self.handle_file_operations(command)
            
            # Search and information
            elif any(word in command_lower for word in ['search', 'find', 'look up', 'what is', 'who is']):
                return await self.handle_search(command)
            
            elif any(word in command_lower for word in ['weather', 'temperature']):
                return await self.handle_search(f"weather today {command}")
            
            elif any(word in command_lower for word in ['news', 'latest news']):
                return await self.handle_search("latest news today")
            
            # Fun and social
            elif 'joke' in command_lower:
                jokes = [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                    "Why did the scarecrow win an award? He was outstanding in his field!",
                    "I'm reading a book about anti-gravity. It's impossible to put down!",
                    "Why don't eggs tell jokes? They'd crack each other up!"
                ]
                import random
                return random.choice(jokes)
            
            elif any(word in command_lower for word in ['thank you', 'thanks']):
                return "You're welcome! Happy to help with anything else - search, images, music, content creation, you name it!"
            
            elif any(word in command_lower for word in ['goodbye', 'bye', 'see you']):
                return "Goodbye! Remember, I'm always here for voice commands, image creation, music, and more!"
            
            elif any(word in command_lower for word in ['help', 'what can you do', 'capabilities']):
                return self.get_capabilities_summary()
            
            else:
                # Use chatbot for general conversation
                if self.chatbot:
                    return await self.handle_chat(command)
                else:
                    return "I heard you! Try commands like: 'create an image', 'play music', 'search for something', 'write content', or 'what can you do?'"
        
        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
            return "Sorry, I encountered an error processing your request."
    
    async def handle_search(self, query: str) -> str:
        """Handle search requests"""
        try:
            if not self.search_engine:
                return "Search is not available right now."
            
            # Remove common prefixes
            clean_query = query.lower()
            for prefix in ['search for', 'find', 'look up', 'what is', 'who is']:
                clean_query = clean_query.replace(prefix, '').strip()
            
            # Use GoogleSearch function directly
            search_result = await asyncio.to_thread(GoogleSearch, clean_query)
            
            # Extract useful information from the search result
            if "No search results found" in search_result:
                return f"I couldn't find information about '{clean_query}'. Try rephrasing your question."
            
            # Parse the search result and extract first meaningful result
            if "[start]" in search_result and "[end]" in search_result:
                start_idx = search_result.find("[start]") + 7
                end_idx = search_result.find("[end]")
                content = search_result[start_idx:end_idx].strip()
                
                # Extract first result
                lines = content.split('\n')
                first_result = ""
                for line in lines:
                    if "Description:" in line:
                        first_result = line.replace("📃 Description:", "").strip()
                        break
                
                if first_result:
                    return f"Here's what I found: {first_result}"
                else:
                    return f"I found some information about '{clean_query}' but couldn't extract a clear answer."
            
            return "I had trouble processing the search results."
        
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return "Sorry, I had trouble searching for that information."
    
    async def handle_chat(self, message: str) -> str:
        """Handle general chat with the chatbot"""
        try:
            if not self.chatbot:
                return "Chat functionality is not available."
            
            # Use chatbot for response
            response = await asyncio.to_thread(self.chatbot.get_response, message)
            
            # Clean up response
            if isinstance(response, dict):
                response = response.get('response', str(response))
            
            return str(response).strip()
        
        except Exception as e:
            self.logger.error(f"Chat error: {e}")
            return "I'm thinking... but having trouble generating a response right now."
    
    async def handle_image_generation(self, command: str) -> str:
        """Handle image generation requests"""
        try:
            if not IMAGE_GENERATION_AVAILABLE:
                return "Image generation is not available. The required AI models aren't installed."
            
            # Extract the image prompt
            prompt = command.lower()
            for prefix in ['create image', 'generate image', 'make image', 'draw']:
                prompt = prompt.replace(prefix, '').strip()
            
            if not prompt:
                return "Please tell me what image you'd like me to create. For example: 'create image of a sunset over mountains'"
            
            # Create images folder if it doesn't exist
            images_dir = Path("Generated_Images")
            images_dir.mkdir(exist_ok=True)
            
            # Generate image in a background thread
            def generation_task():
                try:
                    image_generator = EnhancedTextToImage(
                        prompt=prompt,
                        workspace=str(images_dir),
                        num_images=1,
                        enhance_images=True,
                        guidance_scale=8.5,
                        num_inference_steps=75,
                        use_xl=True
                    )
                    image_generator.run()
                    self.logger.info(f"Image generation complete for prompt: '{prompt}'")
                except Exception as e:
                    self.logger.error(f"Image generation thread error: {e}")

            # Run the generation task in a background thread
            threading.Thread(target=generation_task, daemon=True).start()
            
            # Return immediately
            return f"I'm generating an image of '{prompt}' for you. This may take a few minutes. The image will be saved in the Generated_Images folder."
        
        except Exception as e:
            self.logger.error(f"Image generation error: {e}")
            return "Sorry, I had trouble generating the image. Make sure you have the required AI models installed."
    
    async def handle_music(self, command: str) -> str:
        """Handle music playback requests"""
        try:
            if not AUTOMATION_AVAILABLE:
                return "Music functionality is not available."
            
            # Use the PlayMusic function from automation
            success = await asyncio.to_thread(PlayMusic, command)
            
            if success:
                return "I've started playing music for you!"
            else:
                return "I had trouble playing music. Make sure you have Spotify or Apple Music installed."
        
        except Exception as e:
            self.logger.error(f"Music playback error: {e}")
            return "Sorry, I couldn't play music right now."
    
    async def handle_youtube(self, command: str) -> str:
        """Handle YouTube requests"""
        try:
            if not AUTOMATION_AVAILABLE:
                return "YouTube functionality is not available."
            
            # Extract search term
            search_term = command.lower()
            for prefix in ['youtube', 'play video', 'search youtube']:
                search_term = search_term.replace(prefix, '').strip()
            
            if not search_term:
                return "What would you like to search for on YouTube?"
            
            if 'play' in command.lower():
                # Play video directly
                success = await asyncio.to_thread(PlayYoutube, search_term)
                return f"Playing '{search_term}' on YouTube!"
            else:
                # Search YouTube
                success = await asyncio.to_thread(YouTubeSearch, search_term)
                return f"I've opened YouTube search results for '{search_term}'"
        
        except Exception as e:
            self.logger.error(f"YouTube error: {e}")
            return "Sorry, I had trouble with YouTube."
    
    async def handle_content_creation(self, command: str) -> str:
        """Handle content creation requests"""
        try:
            if not AUTOMATION_AVAILABLE:
                return "Content creation functionality is not available."
            
            # Extract content topic
            topic = command.lower()
            for prefix in ['write', 'content', 'create content', 'letter', 'essay', 'poem']:
                topic = topic.replace(prefix, '').strip()
            
            if not topic:
                return "What would you like me to write? For example: 'write a poem about nature' or 'create content about artificial intelligence'"
            
            # Use the Content function from automation
            success = await asyncio.to_thread(Content, f"Content {topic}")
            
            if success:
                return f"I've created content about '{topic}' and saved it to a file. The file should open automatically!"
            else:
                return f"I had trouble creating content about '{topic}'."
        
        except Exception as e:
            self.logger.error(f"Content creation error: {e}")
            return "Sorry, I had trouble creating content."
    
    async def handle_data_analysis(self, command: str) -> str:
        """Handle data analysis requests with comprehensive automatic analysis"""
        try:
            if not DATA_ANALYZER_AVAILABLE:
                return "Data analysis functionality is not available. Please install required dependencies."
            
            # Extract file path or prompt for data
            prompt = command.lower()
            for prefix in ['analyze data', 'analyze my data', 'run analytics', 'data analysis', 'analyze file']:
                prompt = prompt.replace(prefix, '').strip()
            
            # Look for common data files in current directory
            data_files = []
            for ext in ['.csv', '.xlsx', '.xls']:
                data_files.extend(Path('.').glob(f'*{ext}'))
            
            if not data_files and not prompt:
                return "I can analyze your data! Please put a CSV or Excel file in the current directory, or tell me the file path. For example: 'analyze data from sales.csv'"
            
            selected_file = None
            
            if prompt:
                # Try to find the specified file
                for file in data_files:
                    if prompt in str(file).lower():
                        selected_file = file
                        break
                
                if not selected_file:
                    return f"I couldn't find a data file matching '{prompt}'. Available files: {', '.join([f.name for f in data_files]) if data_files else 'None found'}"
            
            elif data_files:
                # Use the first available data file
                selected_file = data_files[0]
            else:
                return "I couldn't find any data files to analyze. Please ensure you have CSV or Excel files in the directory."
            
            # Run comprehensive automatic analysis
            self.logger.info(f"Starting comprehensive automatic analysis for: {selected_file.name}")
            
            # Create analyzer instance
            analyzer = UniversalAutonomousResearchAnalytics()
            
            # Run complete automatic analysis with all recommended steps
            print(f"\n🚀 Starting comprehensive analysis of {selected_file.name}...")
            results = await asyncio.to_thread(
                analyzer.run_complete_automatic_analysis, 
                str(selected_file), 
                f"analysis_results_{selected_file.stem}"
            )
            
            if results["success"]:
                # Generate summary response
                confidence = "High" if results["model_performance"] > 0.7 else "Moderate" if results["model_performance"] > 0.4 else "Limited"
                
                summary_response = f"""
✅ Complete analysis of '{selected_file.name}' finished successfully!

📊 ANALYSIS SUMMARY:
• Research Focus: {results['target_variable']}
• Predictive Accuracy: {results['model_performance']:.2%} ({confidence} confidence)
• Visualizations Created: {results['eda_plots_count']} comprehensive plots
• Business Recommendations: {len(results['recommendations'])} actionable insights

📁 RESULTS OPENED:
• Interactive HTML summary opened in your browser
• Analysis folder opened in Finder
• Key visualizations displayed
• Comprehensive reports generated

🎯 TOP RECOMMENDATIONS:"""
                
                # Add top 3 recommendations
                for i, rec in enumerate(results['recommendations'][:3], 1):
                    summary_response += f"\n{i}. {rec}"
                
                summary_response += f"\n\n📁 All results saved to: {results['output_dir']}"
                summary_response += "\nThe complete analysis includes detailed reports, visualizations, statistical insights, and business recommendations - everything is now open and ready to review!"
                
                return summary_response
            else:
                error_msg = results.get('error', 'Unknown error occurred')
                return f"I encountered an issue during analysis: {error_msg}. Please check that the data file is valid and accessible."
        
        except Exception as e:
            self.logger.error(f"Data analysis error: {e}")
            return "Sorry, I had trouble with the comprehensive data analysis. Make sure you have the required dependencies installed and the data file is accessible."
    
    async def handle_3d_model_generation(self, command: str) -> str:
        """Handle 3D model generation requests"""
        try:
            if not MODEL_MAKER_AVAILABLE:
                return "3D model generation is not available. Please install required dependencies like shap-e and trimesh."

            # Extract the 3D model prompt
            prompt = command.lower()
            for prefix in ['create 3d model', 'generate 3d', 'make 3d model', '3d model']:
                prompt = prompt.replace(prefix, '').strip()

            if not prompt:
                return "Please tell me what 3D model you'd like me to create. For example: 'create 3D model of a chair' or 'generate 3D car'"

            # Create output directory
            models_dir = Path("Generated_3D_Models")
            models_dir.mkdir(exist_ok=True)

            # Generate 3D model in a background thread
            def generation_task():
                try:
                    self.logger.info(f"Starting 3D model generation for prompt: '{prompt}'")
                    model_generator = ModelMaker.EnhancedModelGenerator()
                    config = ModelMaker.GenerationConfig(quality_preset="balanced", auto_preview=True)
                    
                    # Auto-generate output path
                    clean_name = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    clean_name = clean_name.replace(' ', '_')
                    output_path = models_dir / f"{clean_name}.obj"

                    success, message = model_generator.generate_enhanced_model(
                        text_prompt=prompt,
                        output_path=str(output_path),
                        config=config
                    )
                    
                    if success:
                        self.logger.info(f"3D model generation successful: {message}")
                    else:
                        self.logger.error(f"3D model generation failed: {message}")

                except Exception as e:
                    self.logger.error(f"3D model generation thread error: {e}", exc_info=True)

            # Run the generation task in a background thread
            threading.Thread(target=generation_task, daemon=True).start()

            # Return immediately
            return f"I'm generating a 3D model of '{prompt}' for you. This may take a few minutes. The model will be saved in the 'Generated_3D_Models' folder and should open automatically."

        except Exception as e:
            self.logger.error(f"3D model generation error: {e}", exc_info=True)
            return "Sorry, I had trouble generating the 3D model. Make sure you have the required 3D modeling dependencies installed."
    
    async def handle_agent_task(self, command: str) -> str:
        """Handle AI agent task submission"""
        try:
            if not self.agent_system:
                return "Multi-agent system is not available or not initialized correctly."

            prompt = command.lower()
            # A more robust way to remove prefixes
            prefixes_to_remove = ['create agent to', 'deploy agent to', 'agent team to', 'create ai agent to', 'have an agent to', 'ask an agent to', 'create agent', 'deploy agent', 'agent team', 'create ai agent']
            for prefix in sorted(prefixes_to_remove, key=len, reverse=True):
                if prompt.startswith(prefix):
                    prompt = prompt[len(prefix):].strip()
                    break
            
            # Determine required capabilities from prompt
            required_capabilities = []
            if any(word in prompt for word in ['research', 'find', 'investigate']):
                required_capabilities.append('research')
            if any(word in prompt for word in ['analyze', 'data', 'statistics']):
                required_capabilities.append('analysis')
            if any(word in prompt for word in ['code', 'program', 'develop', 'implement']):
                required_capabilities.append('programming')
            if any(word in prompt for word in ['validate', 'verify', 'check']):
                required_capabilities.append('validation')
            if any(word in prompt for word in ['optimize', 'improve', 'enhance']):
                required_capabilities.append('optimization')

            # Submit the task
            task_id = await self.agent_system.submit_task(
                description=prompt,
                required_capabilities=required_capabilities
            )
            
            return f"I've submitted your task to the agent system with ID: {task_id}. The agents are now working on it. You can ask for the status later."
        
        except Exception as e:
            self.logger.error(f"Agent task submission error: {e}", exc_info=True)
            return "Sorry, I had trouble submitting the task to the agent system."
    
    async def handle_get_task_status(self, command: str) -> str:
        """Handles checking the status of a task in the agent system."""
        try:
            if not self.agent_system:
                return "Multi-agent system is not available or not initialized correctly."

            # Extract task ID from command
            parts = command.split()
            task_id = None
            for part in parts:
                # Basic check for UUID-like string
                if '-' in part and len(part) > 20:
                    task_id = part
                    break
            
            if not task_id:
                return "I couldn't find a task ID in your request. Please specify the task ID, for example: 'what is the status of task 123e4567-e89b-12d3-a456-426614174000'"

            status_data = await self.agent_system.get_task_status(task_id)

            if not status_data:
                return f"I couldn't find any information for task ID {task_id}."

            status = status_data.get('status', 'Unknown').replace('_', ' ')
            result = status_data.get('result')
            error = status_data.get('error')

            response = f"The status of task {task_id} is {status}."
            if status == 'completed' and result:
                response += f" The result is: {result}"
            elif status == 'failed' and error:
                response += f" It failed with an error: {error}"

            return response

        except Exception as e:
            self.logger.error(f"Agent task status error: {e}", exc_info=True)
            return "Sorry, I had trouble getting the task status."
    
    async def handle_system_updates(self, command: str) -> str:
        """Handle system updates and self-improvement"""
        try:
            if not AUTO_UPDATER_AVAILABLE:
                return "System update functionality is not available. Please install required dependencies."
            
            # Extract update type
            prompt = command.lower()
            
            # Create updater instance
            updater = IntelligentAIUpdater()
            
            if any(word in prompt for word in ['backup', 'backup system']):
                # Create system backup
                backup_path = updater.create_backup()
                return f"I've created a complete backup of the current system state at: {backup_path}. This includes all capabilities, configurations, and learned knowledge."
            
            elif any(word in prompt for word in ['update', 'learn new features']):
                # Analyze current capabilities
                capabilities = updater.current_capabilities
                num_functions = len(capabilities.get('functions', {}))
                num_classes = len(capabilities.get('classes', {}))
                num_modules = len(capabilities.get('modules', {}))
                
                return f"I've analyzed my current capabilities: {num_functions} functions, {num_classes} classes across {num_modules} modules. I'm ready to learn new features! To add new capabilities, provide me with code or feature descriptions and I'll intelligently integrate them while avoiding conflicts."
            
            else:
                # General system status
                return "I can help you with system management! I can 'backup system' to create a safety copy, 'update yourself' to analyze and learn new features, or provide system status information. What would you like me to do?"
        
        except Exception as e:
            self.logger.error(f"System update error: {e}")
            return "Sorry, I had trouble with system updates. Make sure you have the required auto-updater dependencies installed."
    
    # ==================== SYSTEM CONTROL FUNCTIONS ====================
    
    async def handle_app_control(self, command: str) -> str:
        """Handle application opening and control"""
        try:
            command_lower = command.lower().strip()
            
            # Extract app name from command
            app_name = ""
            for prefix in ['open', 'launch', 'start', 'run']:
                if prefix in command_lower:
                    app_name = command_lower.replace(prefix, '').strip()
                    break
            
            if not app_name:
                return "Please tell me which application you'd like to open. For example: 'open Spotify' or 'launch Chrome'"
            
            # Common application mappings
            app_mappings = {
                'spotify': 'Spotify',
                'chrome': 'Google Chrome',
                'browser': 'Google Chrome',
                'safari': 'Safari',
                'finder': 'Finder',
                'terminal': 'Terminal',
                'calculator': 'Calculator',
                'calendar': 'Calendar',
                'notes': 'Notes',
                'mail': 'Mail',
                'messages': 'Messages',
                'facetime': 'FaceTime',
                'photos': 'Photos',
                'music': 'Music',
                'app store': 'App Store',
                'system preferences': 'System Preferences',
                'activity monitor': 'Activity Monitor',
                'textedit': 'TextEdit',
                'preview': 'Preview',
                'quicktime': 'QuickTime Player',
                'vlc': 'VLC media player',
                'discord': 'Discord',
                'slack': 'Slack',
                'zoom': 'zoom.us',
                'teams': 'Microsoft Teams',
                'vs code': 'Visual Studio Code',
                'code': 'Visual Studio Code',
                'xcode': 'Xcode',
                'word': 'Microsoft Word',
                'excel': 'Microsoft Excel',
                'powerpoint': 'Microsoft PowerPoint',
                'keynote': 'Keynote',
                'pages': 'Pages',
                'numbers': 'Numbers'
            }
            
            # Get the proper app name
            actual_app_name = app_mappings.get(app_name, app_name.title())
            
            # Use AppleScript to open the application
            success = await self._execute_applescript(f'tell application "{actual_app_name}" to activate')
            
            if success:
                return f"I've opened {actual_app_name} for you!"
            else:
                # Try alternative method using 'open' command
                success = await self._execute_system_command(f'open -a "{actual_app_name}"')
                if success:
                    return f"I've opened {actual_app_name} for you!"
                else:
                    return f"I couldn't find or open {actual_app_name}. Make sure it's installed on your system."
        
        except Exception as e:
            self.logger.error(f"App control error: {e}")
            return "Sorry, I had trouble opening that application."
    
    async def handle_system_control(self, command: str) -> str:
        """Handle system control commands (volume, brightness, wifi, etc.)"""
        try:
            command_lower = command.lower().strip()
            
            # Volume Control
            if any(word in command_lower for word in ['volume up', 'increase volume', 'louder']):
                success = await self._execute_applescript('set volume output volume (output volume of (get volume settings) + 10)')
                return "Volume increased!" if success else "I couldn't adjust the volume."
            
            elif any(word in command_lower for word in ['volume down', 'decrease volume', 'quieter', 'lower volume']):
                success = await self._execute_applescript('set volume output volume (output volume of (get volume settings) - 10)')
                return "Volume decreased!" if success else "I couldn't adjust the volume."
            
            elif any(word in command_lower for word in ['mute', 'silence']):
                success = await self._execute_applescript('set volume with output muted')
                return "System muted!" if success else "I couldn't mute the system."
            
            elif any(word in command_lower for word in ['unmute', 'sound on']):
                success = await self._execute_applescript('set volume without output muted')
                return "System unmuted!" if success else "I couldn't unmute the system."
            
            # Brightness Control
            elif any(word in command_lower for word in ['brightness up', 'brighter', 'increase brightness']):
                success = await self._execute_system_command('brightness 1')  # Increase by 1 level
                return "Brightness increased!" if success else "I couldn't adjust brightness. You may need to install the 'brightness' tool."
            
            elif any(word in command_lower for word in ['brightness down', 'dimmer', 'decrease brightness']):
                success = await self._execute_system_command('brightness -1')  # Decrease by 1 level
                return "Brightness decreased!" if success else "I couldn't adjust brightness. You may need to install the 'brightness' tool."
            
            # WiFi Control
            elif any(word in command_lower for word in ['wifi on', 'turn on wifi', 'enable wifi']):
                success = await self._execute_system_command('networksetup -setairportpower en0 on')
                return "WiFi turned on!" if success else "I couldn't control WiFi."
            
            elif any(word in command_lower for word in ['wifi off', 'turn off wifi', 'disable wifi']):
                success = await self._execute_system_command('networksetup -setairportpower en0 off')
                return "WiFi turned off!" if success else "I couldn't control WiFi."
            
            # Bluetooth Control
            elif any(word in command_lower for word in ['bluetooth on', 'turn on bluetooth', 'enable bluetooth']):
                success = await self._execute_system_command('blueutil -p 1')
                return "Bluetooth turned on!" if success else "I couldn't control Bluetooth. You may need to install 'blueutil'."
            
            elif any(word in command_lower for word in ['bluetooth off', 'turn off bluetooth', 'disable bluetooth']):
                success = await self._execute_system_command('blueutil -p 0')
                return "Bluetooth turned off!" if success else "I couldn't control Bluetooth. You may need to install 'blueutil'."
            
            # System Sleep/Lock
            elif any(word in command_lower for word in ['sleep', 'put computer to sleep']):
                success = await self._execute_applescript('tell application "System Events" to sleep')
                return "Putting the computer to sleep..." if success else "I couldn't put the computer to sleep."
            
            elif any(word in command_lower for word in ['lock screen', 'lock computer']):
                success = await self._execute_system_command('pmset displaysleepnow')
                return "Locking the screen..." if success else "I couldn't lock the screen."
            
            # Desktop/Dock Control
            elif any(word in command_lower for word in ['hide dock', 'dock hide']):
                success = await self._execute_system_command('defaults write com.apple.dock autohide -bool true && killall Dock')
                return "Dock hidden!" if success else "I couldn't hide the dock."
            
            elif any(word in command_lower for word in ['show dock', 'dock show']):
                success = await self._execute_system_command('defaults write com.apple.dock autohide -bool false && killall Dock')
                return "Dock shown!" if success else "I couldn't show the dock."
            
            # Mission Control & Spaces
            elif any(word in command_lower for word in ['mission control', 'show all windows']):
                success = await self._execute_applescript('tell application "System Events" to key code 160')
                return "Opening Mission Control..." if success else "I couldn't open Mission Control."
            
            elif any(word in command_lower for word in ['desktop', 'show desktop']):
                success = await self._execute_applescript('tell application "System Events" to key code 103')
                return "Showing desktop..." if success else "I couldn't show the desktop."
            
            # Trash Control
            elif any(word in command_lower for word in ['empty trash', 'clear trash']):
                success = await self._execute_applescript('tell application "Finder" to empty the trash')
                return "Emptying trash..." if success else "I couldn't empty the trash."
            
            else:
                return "I can help you control your system! Try commands like: 'volume up', 'brightness down', 'wifi on', 'bluetooth off', 'sleep', 'lock screen', 'hide dock', 'mission control', 'empty trash', or 'open [app name]'."
        
        except Exception as e:
            self.logger.error(f"System control error: {e}")
            return "Sorry, I had trouble with that system control command."
    
    async def handle_file_operations(self, command: str) -> str:
        """Handle file and folder operations"""
        try:
            command_lower = command.lower().strip()
            
            # Open folders
            if any(word in command_lower for word in ['open downloads', 'downloads folder']):
                success = await self._execute_system_command('open ~/Downloads')
                return "Opening Downloads folder..." if success else "I couldn't open Downloads."
            
            elif any(word in command_lower for word in ['open documents', 'documents folder']):
                success = await self._execute_system_command('open ~/Documents')
                return "Opening Documents folder..." if success else "I couldn't open Documents."
            
            elif any(word in command_lower for word in ['open desktop', 'desktop folder']):
                success = await self._execute_system_command('open ~/Desktop')
                return "Opening Desktop folder..." if success else "I couldn't open Desktop."
            
            elif any(word in command_lower for word in ['open home', 'home folder']):
                success = await self._execute_system_command('open ~')
                return "Opening Home folder..." if success else "I couldn't open Home folder."
            
            elif any(word in command_lower for word in ['open applications', 'applications folder']):
                success = await self._execute_system_command('open /Applications')
                return "Opening Applications folder..." if success else "I couldn't open Applications."
            
            # File operations
            elif any(word in command_lower for word in ['create new folder', 'make folder', 'new folder']):
                # Extract folder name if provided
                folder_name = "New Folder"
                if "called" in command_lower or "named" in command_lower:
                    parts = command_lower.split("called" if "called" in command_lower else "named")
                    if len(parts) > 1:
                        folder_name = parts[1].strip()
                
                success = await self._execute_system_command(f'mkdir -p ~/Desktop/"{folder_name}"')
                return f"Created folder '{folder_name}' on Desktop!" if success else "I couldn't create the folder."
            
            else:
                return "I can help with file operations! Try: 'open downloads', 'open documents', 'open desktop', 'open applications', or 'create new folder called [name]'."
        
        except Exception as e:
            self.logger.error(f"File operations error: {e}")
            return "Sorry, I had trouble with that file operation."
    
    async def _execute_applescript(self, script: str) -> bool:
        """Execute AppleScript command"""
        try:
            result = await asyncio.to_thread(subprocess.run, 
                ['osascript', '-e', script], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"AppleScript execution failed: {e}")
            return False
    
    async def _execute_system_command(self, command: str) -> bool:
        """Execute system command safely"""
        try:
            # Split command safely
            cmd_parts = shlex.split(command)
            result = await asyncio.to_thread(subprocess.run, 
                cmd_parts, 
                capture_output=True, 
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"System command execution failed: {e}")
            return False
    
    def get_capabilities_summary(self) -> str:
        """Return a summary of LEO's capabilities"""
        capabilities = []
        
        # Core capabilities
        capabilities.append("🔍 Search the web and get information")
        capabilities.append("💬 Have intelligent conversations")
        capabilities.append("⏰ Tell you the time and date")
        capabilities.append("🌤️ Get weather information")
        capabilities.append("📰 Find latest news")
        capabilities.append("😄 Tell jokes")
        
        # Enhanced capabilities
        if IMAGE_GENERATION_AVAILABLE:
            capabilities.append("🎨 Generate images from descriptions")
        
        if AUTOMATION_AVAILABLE:
            capabilities.append("🎵 Play music on Spotify or Apple Music")
            capabilities.append("📺 Search and play YouTube videos")
            capabilities.append("📝 Write content, letters, essays, and poems")
        
        if ENHANCED_TTS_AVAILABLE:
            capabilities.append("🔊 High-quality text-to-speech")
        
        summary = "I can help you with: " + ", ".join(capabilities[:5])
        if len(capabilities) > 5:
            summary += f", and {len(capabilities) - 5} more capabilities!"
        
        return summary
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds() / 60  # minutes
        
        return {
            "voice_interactions": self.stats["voice_interactions"],
            "successful_responses": self.stats["successful_responses"],
            "uptime_minutes": round(uptime, 1),
            "components_active": {
                "voice_recognition": SPEECH_RECOGNITION_AVAILABLE,
                "text_to_speech": TTS_AVAILABLE,
                "search_engine": self.search_engine is not None,
                "chatbot": self.chatbot is not None,
                "image_generation": IMAGE_GENERATION_AVAILABLE,
                "automation": AUTOMATION_AVAILABLE,
                "enhanced_tts": ENHANCED_TTS_AVAILABLE
            }
        }

# Global Voice LEO instance and continuous listening control
voice_leo = None
continuous_listening = False
listening_thread = None
gui_widget = None

async def initialize_voice_leo():
    """Initialize the global Voice LEO instance"""
    global voice_leo
    try:
        voice_leo = VoiceLEO()
        success = await voice_leo.initialize()
        
        if success:
            logger.info("🎤 Voice LEO initialized successfully!")
            return True
        else:
            logger.error("❌ Failed to initialize Voice LEO")
            return False
    except Exception as e:
        logger.error(f"❌ Error initializing Voice LEO: {e}")
        return False

def process_voice_input_sync(command: str) -> str:
    """Synchronous wrapper for voice processing (for GUI integration)"""
    global voice_leo
    
    if not voice_leo:
        return "Voice LEO is not initialized."
    
    try:
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(voice_leo.process_voice_command(command))
        loop.close()
        return response
    except Exception as e:
        logger.error(f"Error in sync processing: {e}")
        return "Sorry, I had trouble processing that."

def continuous_listening_loop():
    """Continuous listening loop that runs in background thread"""
    global continuous_listening, gui_widget
    
    print("🎤 Starting continuous listening...")
    
    while continuous_listening:
        try:
            if not voice_leo:
                time.sleep(1)
                continue
            
            # Update GUI status if available
            if gui_widget:
                try:
                    gui_widget.update_status("🎤 LEO is listening...", "#667eea")
                except:
                    pass
            
            print("👂 Listening for voice input...")
            
            # Listen for voice input using enhanced recognition
            command = EnhancedListen()
            
            if command:
                print(f"🎤 Voice command received: '{command}'")
                
                # Update GUI status
                if gui_widget:
                    try:
                        gui_widget.update_status(f"🎤 Processing: '{command[:30]}...'", "#ffa500")
                    except:
                        pass
                
                # Process command
                response = process_voice_input_sync(command)
                
                # Speak response
                Speak(response)
                print(f"🤖 LEO: {response}")
                
                # Update GUI status
                if gui_widget:
                    try:
                        gui_widget.update_status(f"✅ Response sent", "#48bb78")
                        # Reset to listening after 2 seconds
                        from PyQt5.QtCore import QTimer
                        QTimer.singleShot(2000, lambda: gui_widget.update_status("🎤 Ready - Say something...", "#667eea") if gui_widget else None)
                    except:
                        pass
                
                # Update stats
                if voice_leo:
                    voice_leo.stats["successful_responses"] += 1
            
            else:
                # Short pause when no speech detected to avoid excessive CPU usage
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("🛑 Continuous listening stopped by user")
            break
        except Exception as e:
            logger.error(f"Continuous listening error: {e}")
            time.sleep(1)
    
    print("🛑 Continuous listening ended")

def start_continuous_listening(widget=None):
    """Start continuous listening in background thread"""
    global continuous_listening, listening_thread, gui_widget
    
    if continuous_listening:
        return
    
    gui_widget = widget
    continuous_listening = True
    
    listening_thread = threading.Thread(target=continuous_listening_loop, daemon=True)
    listening_thread.start()
    
    logger.info("🎤 Continuous listening started")
    print("🎤 Voice LEO is now listening continuously...")
    print("💬 Just start talking - no need to click buttons!")

def stop_continuous_listening():
    """Stop continuous listening"""
    global continuous_listening, listening_thread
    
    continuous_listening = False
    
    if listening_thread:
        listening_thread.join(timeout=2)
    
    logger.info("🛑 Continuous listening stopped")

def enhanced_voice_handler():
    """Enhanced voice interaction handler for GUI integration (fallback)"""
    try:
        # Initialize Voice LEO if not already done
        if not voice_leo:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(initialize_voice_leo())
            loop.close()
            
            if not success:
                Speak("Voice system initialization failed.")
                return "Initialization failed"
        
        # Listen for voice input (single interaction) using enhanced recognition
        command = EnhancedListen()
        
        if command:
            print(f"🎤 Voice command received: '{command}'")
            
            # Process command
            response = process_voice_input_sync(command)
            
            # Speak response
            Speak(response)
            
            # Update stats
            if voice_leo:
                voice_leo.stats["successful_responses"] += 1
            
            return response
        else:
            return "No speech detected"
            
    except Exception as e:
        logger.error(f"Voice handler error: {e}")
        Speak("Sorry, I had trouble with voice processing.")
        return "Voice processing error"

# Monkey patch the GUI to use our enhanced voice handler
def patch_gui_voice_functionality():
    """Patch GUI.py to use our enhanced voice processing"""
    try:
        import Frontend.GUI as gui_module
        
        # Store original methods
        original_activate_voice = gui_module.ElegantMainWidget.activate_voice
        original_start_voice_recognition = gui_module.ElegantMainWidget.start_voice_recognition
        original_init = gui_module.ElegantMainWidget.__init__
        
        def enhanced_init(self, main_window=None):
            """Enhanced initialization with auto-start continuous listening"""
            # Call original initialization
            original_init(self, main_window)
            
            # Auto-start continuous listening after GUI is ready
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self.auto_start_continuous_listening())
        
        def auto_start_continuous_listening(self):
            """Automatically start continuous listening"""
            try:
                # Start continuous listening with this widget as reference
                start_continuous_listening(self)
                
                # Update status to show always listening
                self.update_status("🎤 Always Listening - Just speak!", "#667eea")
                
                # Welcome message
                if TTS_AVAILABLE:
                    Speak("Voice LEO is ready and listening. Just start talking!")
                else:
                    print("[LEO]: Voice LEO is ready and listening. Just start talking!")
                    
            except Exception as e:
                logger.error(f"Failed to start continuous listening: {e}")
                self.update_status("⚠️ Voice setup incomplete", "#e53e3e")
        
        def enhanced_activate_voice(self):
            """Enhanced voice activation - now just shows status since we're always listening"""
            if continuous_listening:
                self.update_status("🎤 Already listening - just speak!", "#48bb78")
                
                # Quick response to show it's working
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(2000, lambda: self.update_status("🎤 Always Listening - Just speak!", "#667eea"))
            else:
                # Fallback to manual activation if continuous listening failed
                self.update_status("🎤 LEO is listening...", "#667eea")
                
                if TTS_AVAILABLE:
                    Speak("I'm listening...")
                
                # Use our enhanced voice handler
                try:
                    response = enhanced_voice_handler()
                    if response and response not in ["No speech detected", "Voice processing error"]:
                        self.update_status(f"🎤 LEO: Processing complete", "#48bb78")
                    else:
                        self.update_status("🎤 No speech detected", "#ed8936")
                except Exception as e:
                    self.update_status("🎤 Voice error occurred", "#e53e3e")
                    logger.error(f"Enhanced voice activation error: {e}")
                
                # Reset status
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(3000, lambda: self.update_status("Ready to assist you", "#4a5568"))
        
        # Apply patches
        gui_module.ElegantMainWidget.__init__ = enhanced_init
        gui_module.ElegantMainWidget.auto_start_continuous_listening = auto_start_continuous_listening
        gui_module.ElegantMainWidget.activate_voice = enhanced_activate_voice
        
        logger.info("✅ GUI voice functionality enhanced with continuous listening!")
        
    except Exception as e:
        logger.error(f"⚠️ Could not patch GUI functionality: {e}")

def main():
    """Main entry point for Voice LEO with GUI"""
    print("\n" + "="*60)
    print("🎤 VOICE LEO - AI ASSISTANT")
    print("🗣️  Voice-Focused • 🖥️  GUI Integrated • 🧠 Intelligent")
    print("="*60)
    
    try:
        # Initialize Voice LEO backend
        print("🔧 Initializing Voice LEO backend...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(initialize_voice_leo())
        loop.close()
        
        if not success:
            print("❌ Failed to initialize Voice LEO backend")
            return
        
        # Patch GUI for enhanced functionality
        print("🔧 Enhancing GUI with Voice LEO...")
        patch_gui_voice_functionality()
        
        # Show comprehensive system status
        if voice_leo:
            stats = voice_leo.get_stats()
            print(f"\n📊 Enhanced Voice LEO Status:")
            print(f"   • Voice Recognition: {'✅' if stats['components_active']['voice_recognition'] else '❌'}")
            print(f"   • Text-to-Speech: {'✅' if stats['components_active']['text_to_speech'] else '❌'}")
            print(f"   • Search Engine: {'✅' if stats['components_active']['search_engine'] else '❌'}")
            print(f"   • Chatbot: {'✅' if stats['components_active']['chatbot'] else '❌'}")
            
            # Show Master Integration System status
            if voice_leo.master_system:
                master_status = voice_leo.master_system.get_system_status()
                print(f"\n🌟 Master Integration System:")
                print(f"   • Status: {master_status['status'].upper()} ✅")
                print(f"   • Mode: {master_status['mode'].replace('_', ' ').title()}")
                print(f"   • Active Components: {len(master_status['active_components'])}")
                for component in master_status['active_components']:
                    print(f"     - {component.replace('_', ' ').title()}")
            else:
                print(f"\n⚠️ Master Integration System: Not Available")
            
            # Show enhanced capabilities status
            print(f"\n🚀 Enhanced Capabilities:")
            print(f"   • Image Generation: {'✅' if IMAGE_GENERATION_AVAILABLE else '❌'}")
            print(f"   • Music/YouTube: {'✅' if AUTOMATION_AVAILABLE else '❌'}")
            print(f"   • Data Analysis: {'✅' if DATA_ANALYZER_AVAILABLE else '❌'}")
            print(f"   • 3D Model Creation: {'✅' if MODEL_MAKER_AVAILABLE else '❌'}")
            print(f"   • Multi-Agent System: {'✅' if AGENT_SYSTEM_AVAILABLE else '❌'}")
            print(f"   • Auto-Updater: {'✅' if AUTO_UPDATER_AVAILABLE else '❌'}")
        
        print(f"\n💬 Enhanced Voice Commands:")
        print(f"   🗣️  Basic: 'Hello', 'What time is it?', 'Tell me a joke'")
        print(f"   🔍 Search: 'Search for [topic]', 'What's the weather?', 'Latest news'")
        print(f"   🎨 Create: 'Create image of [description]', 'Write content about [topic]'")
        print(f"   🎵 Media: 'Play music', 'YouTube [search]', 'Open Spotify'")
        print(f"   💻 System: 'Volume up', 'Brightness down', 'WiFi on', 'Sleep'")
        print(f"   📂 Files: 'Open Downloads', 'Create folder called [name]'")
        print(f"   🖥️  Apps: 'Open [any app]', 'Launch Chrome', 'Start Calculator'")
        
        print(f"\n🎤 Starting Voice LEO GUI...")
        print("="*60)
        
        # Start the GUI
        GraphicalUserInterface()
        
    except KeyboardInterrupt:
        print("\n🛑 Voice LEO stopped by user")
    except Exception as e:
        print(f"❌ Voice LEO failed to start: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
