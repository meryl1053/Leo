#!/usr/bin/env python3
"""
🚀 LEO - INTEGRATED AI ASSISTANT 🚀
Unified entry point for the complete LEO AI system using the Master Integration System.

This new main file brings together ALL components:
- Advanced Learning Engine (continuous learning)
- Intelligent Resource Optimizer (GPU/CPU optimization)
- Multimodal AI Processor (text, voice, image, 3D)
- Ultimate AI Orchestrator (task coordination)
- All existing LEO components (chatbot, search, voice, etc.)

Features:
- Voice activation and conversation
- Multimodal processing
- Real-time learning and adaptation
- Intelligent resource management
- Comprehensive monitoring
- Cross-component integration
"""

import asyncio
import logging
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import traceback
import threading
import signal

# Add Backend to path for imports
sys.path.append(str(Path(__file__).parent / "Backend"))
sys.path.append(str(Path(__file__).parent / "Frontend"))

# Import the Master Integration System
try:
    from Backend.MasterIntegrationSystem import MasterIntegrationSystem, SystemMode
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Master Integration System not available: {e}")
    INTEGRATION_AVAILABLE = False

# Import GUI components for voice interaction
try:
    from Frontend.GUI import Listen, Speak
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  GUI components not available: {e}")
    GUI_AVAILABLE = False

# Import enhanced Voice LEO system
try:
    from Voice_LEO import (
        initialize_voice_leo, 
        process_voice_input_sync, 
        enhanced_voice_handler,
        voice_leo,
        VoiceLEO
    )
    VOICE_LEO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Enhanced Voice LEO not available: {e}")
    VOICE_LEO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('leo_integrated.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LEOAssistant:
    """Main LEO Assistant class with full integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.master_system = None
        self.current_user = "user_default"
        self.interaction_count = 0
        self.start_time = datetime.now()
        
        # Configuration
        self.config = {
            "voice_enabled": True,
            "continuous_listening": True,
            "learning_enabled": True,
            "resource_optimization": True,
            "multimodal_processing": True,
            "system_monitoring": True
        }
        
        # Statistics
        self.stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "average_response_time": 0.0,
            "learning_improvements": 0
        }
    
    async def initialize(self):
        """Initialize the complete LEO system"""
        try:
            self.logger.info("🚀 Initializing LEO Integrated AI Assistant...")
            
            if not INTEGRATION_AVAILABLE:
                self.logger.error("❌ Master Integration System not available!")
                return False
            
            # Initialize Master Integration System
            self.master_system = MasterIntegrationSystem()
            await self.master_system.initialize()
            
            # Set system mode based on configuration
            if self.config["voice_enabled"]:
                self.master_system.current_mode = SystemMode.VOICE_ASSISTANT
            else:
                self.master_system.current_mode = SystemMode.CHAT_INTERFACE
            
            self.is_running = True
            
            self.logger.info("✅ LEO system fully initialized and ready!")
            self._print_system_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ LEO initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def _print_system_info(self):
        """Print system information and capabilities"""
        print("\n" + "="*60)
        print("🌟 LEO INTEGRATED AI ASSISTANT")
        print("="*60)
        
        if self.master_system:
            status = self.master_system.get_system_status()
            print(f"📊 System Status: {status['status'].upper()}")
            print(f"🔧 Active Components: {len(status['active_components'])}")
            print(f"⚡ Mode: {status['mode'].replace('_', ' ').title()}")
            
            print(f"\n🧠 Available Capabilities:")
            for component in status['active_components']:
                component_name = component.replace('_', ' ').title()
                print(f"   ✓ {component_name}")
        
        print(f"\n🎯 Configuration:")
        print(f"   • Voice Interaction: {'Enabled' if self.config['voice_enabled'] else 'Disabled'}")
        print(f"   • Continuous Learning: {'Enabled' if self.config['learning_enabled'] else 'Disabled'}")
        print(f"   • Resource Optimization: {'Enabled' if self.config['resource_optimization'] else 'Disabled'}")
        print(f"   • Multimodal Processing: {'Enabled' if self.config['multimodal_processing'] else 'Disabled'}")
        
        print("\n💬 Ready for interaction!")
        if self.config["voice_enabled"]:
            print("🎤 Say something or type 'quit' to exit...")
        else:
            print("⌨️  Type your message or 'quit' to exit...")
        print("="*60 + "\n")
    
    async def run_voice_mode(self):
        """Run LEO in voice interaction mode"""
        try:
            self.logger.info("🎤 Starting voice interaction mode...")
            
            if not GUI_AVAILABLE:
                print("⚠️  Voice components not available, switching to text mode...")
                await self.run_chat_mode()
                return
            
            print("🎤 Voice mode active. Say 'LEO' or just start talking...")
            
            while self.is_running:
                try:
                    # Listen for voice input
                    print("\n👂 Listening...")
                    voice_input = Listen()
                    
                    if voice_input is None:
                        await asyncio.sleep(0.5)
                        continue
                    
                    if voice_input.lower().strip() in ['quit', 'exit', 'goodbye', 'stop']:
                        print("👋 Goodbye!")
                        break
                    
                    print(f"🎤 You said: '{voice_input}'")
                    
                    # Process through integrated system
                    await self._process_interaction(voice_input, "voice")
                    
                except KeyboardInterrupt:
                    print("\n🛑 Voice mode interrupted")
                    break
                except Exception as e:
                    self.logger.error(f"Voice mode error: {e}")
                    print(f"❌ Error in voice processing: {e}")
                    await asyncio.sleep(1)
        
        except Exception as e:
            self.logger.error(f"Voice mode failed: {e}")
            print(f"❌ Voice mode error: {e}")
    
    async def run_chat_mode(self):
        """Run LEO in text chat mode"""
        try:
            self.logger.info("💬 Starting chat interaction mode...")
            
            print("💬 Chat mode active. Type your message and press Enter...")
            
            while self.is_running:
                try:
                    # Get text input
                    user_input = input("\n🤔 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['quit', 'exit', 'goodbye', 'stop']:
                        print("👋 Goodbye!")
                        break
                    
                    # Process through integrated system
                    await self._process_interaction(user_input, "text")
                    
                except KeyboardInterrupt:
                    print("\n🛑 Chat mode interrupted")
                    break
                except EOFError:
                    print("\n👋 Chat ended")
                    break
                except Exception as e:
                    self.logger.error(f"Chat mode error: {e}")
                    print(f"❌ Error in chat processing: {e}")
        
        except Exception as e:
            self.logger.error(f"Chat mode failed: {e}")
            print(f"❌ Chat mode error: {e}")
    
    async def _process_interaction(self, user_input: str, modality: str):
        """Process user interaction through the integrated system"""
        start_time = time.time()
        
        try:
            self.interaction_count += 1
            self.stats["total_interactions"] += 1
            
            print(f"\n🧠 Processing (#{self.interaction_count})...")
            
            # Try enhanced Voice LEO first if available
            if VOICE_LEO_AVAILABLE and modality == "voice":
                try:
                    response = await self._process_with_voice_leo(user_input)
                    if response:
                        processing_time = time.time() - start_time
                        
                        # Display response
                        print(f"🤖 LEO: {response}")
                        
                        # Speak response
                        if GUI_AVAILABLE:
                            try:
                                Speak(response)
                            except Exception as e:
                                self.logger.warning(f"Speech output failed: {e}")
                        
                        print(f"   📍 Processed via: Enhanced Voice LEO")
                        print(f"   ⏱️  Response time: {processing_time:.2f}s")
                        
                        # Update statistics
                        self.stats["successful_interactions"] += 1
                        self.stats["average_response_time"] = (
                            self.stats["average_response_time"] * 0.9 + processing_time * 0.1
                        )
                        
                        return
                        
                except Exception as e:
                    self.logger.warning(f"Voice LEO processing failed, falling back: {e}")
            
            # Process through Master Integration System
            result = await self.master_system.process_request(
                user_id=self.current_user,
                request=user_input,
                modality=modality,
                context={
                    "interaction_number": self.interaction_count,
                    "session_start": self.start_time.isoformat(),
                    "preferred_modality": modality
                }
            )
            
            processing_time = time.time() - start_time
            
            if result.get("success", False):
                response = result.get("response", "I'm processing your request...")
                
                # Display response
                print(f"🤖 LEO: {response}")
                
                # Speak response if in voice mode
                if modality == "voice" and GUI_AVAILABLE:
                    try:
                        Speak(response)
                    except Exception as e:
                        self.logger.warning(f"Speech output failed: {e}")
                
                # Show processing details
                route = result.get("route", [])
                if route:
                    print(f"   📍 Processed via: {' → '.join(route)}")
                
                print(f"   ⏱️  Response time: {processing_time:.2f}s")
                
                # Update statistics
                self.stats["successful_interactions"] += 1
                self.stats["average_response_time"] = (
                    self.stats["average_response_time"] * 0.9 + processing_time * 0.1
                )
                
                # Ask for feedback (occasionally)
                if self.interaction_count % 5 == 0:
                    await self._collect_feedback(result.get("interaction_id"))
                
            else:
                error_msg = result.get("error", "Unknown error occurred")
                print(f"❌ LEO: I encountered an error: {error_msg}")
                print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        
        except Exception as e:
            self.logger.error(f"Interaction processing failed: {e}")
            print(f"❌ LEO: I'm sorry, I encountered an unexpected error.")
            print(f"   Error details: {str(e)}")
    
    async def _process_with_voice_leo(self, user_input: str) -> Optional[str]:
        """Process input using enhanced Voice LEO system"""
        try:
            # Initialize Voice LEO if not already done
            if not voice_leo:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(initialize_voice_leo())
                loop.close()
                
                if not success:
                    return None
            
            # Process with Voice LEO
            response = process_voice_input_sync(user_input)
            return response if response else None
            
        except Exception as e:
            self.logger.error(f"Voice LEO processing error: {e}")
            return None
    
    async def _collect_feedback(self, interaction_id: Optional[str]):
        """Collect user feedback for learning"""
        try:
            if not interaction_id or not self.config["learning_enabled"]:
                return
            
            print("\n📝 Quick feedback (optional):")
            print("   1 = Poor, 2 = Fair, 3 = Good, 4 = Excellent")
            print("   Or just press Enter to skip")
            
            feedback_input = input("   Rating (1-4): ").strip()
            
            if feedback_input and feedback_input.isdigit():
                rating = int(feedback_input)
                if 1 <= rating <= 4:
                    # Convert to satisfaction score (0-1)
                    satisfaction = (rating - 1) / 3.0
                    
                    # Determine feedback type
                    if rating >= 3:
                        feedback_type = "positive"
                    else:
                        feedback_type = "negative"
                    
                    # Provide feedback to learning system
                    await self.master_system.provide_feedback(
                        interaction_id, feedback_type, satisfaction
                    )
                    
                    print(f"   ✅ Thank you for the feedback!")
                    self.stats["learning_improvements"] += 1
        
        except Exception as e:
            self.logger.warning(f"Feedback collection failed: {e}")
    
    def show_statistics(self):
        """Show current session statistics"""
        if not self.is_running:
            return
        
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600
            success_rate = (self.stats["successful_interactions"] / 
                          max(1, self.stats["total_interactions"]))
            
            print(f"\n📊 Session Statistics:")
            print(f"   • Total Interactions: {self.stats['total_interactions']}")
            print(f"   • Success Rate: {success_rate:.1%}")
            print(f"   • Avg Response Time: {self.stats['average_response_time']:.2f}s")
            print(f"   • Learning Feedback: {self.stats['learning_improvements']} provided")
            print(f"   • Session Uptime: {uptime:.1f} hours")
            
            # System status
            if self.master_system:
                system_status = self.master_system.get_system_status()
                print(f"   • System Health: {system_status['status'].upper()}")
                print(f"   • Active Components: {len(system_status['active_components'])}")
        
        except Exception as e:
            self.logger.error(f"Statistics display failed: {e}")
    
    async def run_interactive_mode(self):
        """Run LEO in interactive mode with commands"""
        try:
            print("\n🎮 Interactive mode - Available commands:")
            print("   • 'voice' - Switch to voice mode")
            print("   • 'chat' - Switch to chat mode") 
            print("   • 'status' - Show system status")
            print("   • 'stats' - Show session statistics")
            print("   • 'help' - Show this help")
            print("   • 'quit' - Exit LEO")
            
            while self.is_running:
                try:
                    command = input("\n🎮 Command (or just talk): ").strip().lower()
                    
                    if command in ['quit', 'exit']:
                        break
                    elif command == 'voice':
                        await self.run_voice_mode()
                    elif command == 'chat':
                        await self.run_chat_mode()
                    elif command == 'status':
                        self._show_system_status()
                    elif command == 'stats':
                        self.show_statistics()
                    elif command == 'help':
                        print("\n🎮 Available commands:")
                        print("   • 'voice' - Voice interaction")
                        print("   • 'chat' - Text chat")
                        print("   • 'status' - System status")
                        print("   • 'stats' - Statistics")
                        print("   • 'quit' - Exit")
                    elif command:
                        # Treat as regular input
                        await self._process_interaction(command, "text")
                    
                except KeyboardInterrupt:
                    print("\n🛑 Interactive mode interrupted")
                    break
                except EOFError:
                    print("\n👋 Interactive mode ended")
                    break
        
        except Exception as e:
            self.logger.error(f"Interactive mode failed: {e}")
    
    def _show_system_status(self):
        """Show detailed system status"""
        try:
            if not self.master_system:
                print("❌ Master system not available")
                return
            
            status = self.master_system.get_system_status()
            
            print(f"\n🔍 System Status:")
            print(f"   • System ID: {status['system_id'][:8]}...")
            print(f"   • Status: {status['status'].upper()}")
            print(f"   • Mode: {status['mode'].replace('_', ' ').title()}")
            print(f"   • Uptime: {status['uptime_hours']:.1f} hours")
            print(f"   • Active Users: {status['metrics']['active_users']}")
            
            print(f"\n🧠 Component Health:")
            for name, comp_status in status['component_status'].items():
                health = comp_status['health_score']
                status_icon = "✅" if health > 0.8 else "⚠️" if health > 0.5 else "❌"
                print(f"   {status_icon} {name.replace('_', ' ').title()}: {health:.1%}")
            
            print(f"\n📊 Performance Metrics:")
            print(f"   • Total Requests: {status['metrics']['total_requests']}")
            print(f"   • Success Rate: {status['metrics']['success_rate']:.1%}")
            print(f"   • Avg Response Time: {status['metrics']['average_response_time']:.2f}s")
            print(f"   • Resource Usage: {status['metrics']['resource_utilization']:.1f}%")
        
        except Exception as e:
            self.logger.error(f"Status display failed: {e}")
            print(f"❌ Failed to get system status: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown LEO"""
        try:
            self.logger.info("🛑 Shutting down LEO...")
            self.is_running = False
            
            # Show final statistics
            print("\n" + "="*60)
            print("📊 FINAL SESSION SUMMARY")
            print("="*60)
            self.show_statistics()
            
            # Shutdown master system
            if self.master_system:
                await self.master_system.shutdown()
            
            print("\n✅ LEO shutdown complete. Thank you for using LEO!")
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
            print(f"❌ Shutdown error: {e}")


async def main():
    """Main entry point for LEO"""
    leo = None
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received shutdown signal ({signum})")
        if leo and leo.is_running:
            leo.is_running = False
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🌟 Starting LEO Integrated AI Assistant...")
        
        # Initialize LEO
        leo = LEOAssistant()
        
        if not await leo.initialize():
            print("❌ Failed to initialize LEO")
            return
        
        # Determine run mode
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
        else:
            mode = "interactive"
        
        # Run in specified mode
        if mode == "voice":
            await leo.run_voice_mode()
        elif mode == "chat":
            await leo.run_chat_mode()
        elif mode == "interactive":
            await leo.run_interactive_mode()
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: voice, chat, interactive")
    
    except KeyboardInterrupt:
        print("\n🛑 LEO interrupted by user")
    except Exception as e:
        print(f"❌ LEO encountered an error: {e}")
        traceback.print_exc()
    
    finally:
        if leo:
            await leo.shutdown()


def run_leo():
    """Convenience function to run LEO"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 LEO stopped by user")
    except Exception as e:
        print(f"❌ LEO failed to start: {e}")


if __name__ == "__main__":
    # Print welcome message
    print("\n" + "="*60)
    print("🚀 LEO - INTEGRATED AI ASSISTANT")
    print("🧠 Advanced • 🎤 Voice-Enabled • 📚 Self-Learning")
    print("="*60)
    
    # Run LEO
    run_leo()
