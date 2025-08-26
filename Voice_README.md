# 🎤 Voice LEO - Voice-Only AI Assistant

A streamlined version of LEO focused exclusively on voice interaction with an elegant GUI interface.

## ✨ What's Different

Voice LEO is a simplified version that:
- **🎤 Voice-Only Focus** - Optimized specifically for voice interaction
- **🔄 Always Listening** - Continuous voice recognition (no button clicking!)
- **🖥️ GUI Integrated** - Uses the beautiful existing GUI.py interface
- **⚡ Lightweight** - Fewer dependencies, faster startup
- **🧠 Smart Routing** - Intelligent command processing

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install only the essential packages
pip install -r voice_requirements.txt
```

### 2. Run Voice LEO
```bash
python Voice_LEO.py
```

## 🎯 Features

### Voice Commands Supported:
- **Greetings**: "Hello", "Hi", "Hey"
- **Time/Date**: "What time is it?", "What's today's date?"
- **Search**: "Search for [topic]", "What is [topic]?"
- **Weather**: "What's the weather?", "Temperature today?"
- **News**: "Tell me the news", "Latest news"
- **Entertainment**: "Tell me a joke"
- **Conversation**: Any general chat or questions

### Core Components:
- ✅ **Voice Recognition** - Speech-to-text conversion
- ✅ **Search Engine** - Real-time web search via Google
- ✅ **Chatbot** - Intelligent conversation using Groq AI
- ✅ **GUI Interface** - Beautiful animated interface
- ⚠️ **Text-to-Speech** - Fallback to text display (can be enhanced)

## 🎮 How to Use

1. **Launch** - Run `python Voice_LEO.py`
2. **Wait 2 seconds** - LEO automatically starts listening
3. **Just start talking** - No need to click anything!
4. **See the magic** - LEO processes and responds immediately

### 🔄 **Always Listening Mode**
- Voice LEO starts listening **automatically** when the GUI loads
- Status shows "🎤 Always Listening - Just speak!"
- Simply talk naturally - LEO is always ready
- No button clicks required!

## 🔧 Configuration

Voice LEO automatically:
- Initializes all available components
- Tests API connections
- Shows component status
- Handles errors gracefully

## 📊 What You'll See

```
============================================================
🎤 VOICE LEO - AI ASSISTANT
🗣️  Voice-Focused • 🖥️  GUI Integrated • 🧠 Intelligent
============================================================

📊 Voice LEO Status:
   • Voice Recognition: ✅
   • Text-to-Speech: ❌ (fallback available)
   • Search Engine: ✅
   • Chatbot: ✅

💬 Voice Commands:
   • 'Hello' - Greet LEO
   • 'What time is it?' - Get current time
   • 'Search for [topic]' - Web search
   • 'What's the weather?' - Weather info
   • 'Tell me the news' - Latest news
   • 'Tell me a joke' - Get a joke
   • Any conversation - Chat with LEO
```

## 🆚 Voice LEO vs Full LEO

| Feature | Voice LEO | Full LEO |
|---------|-----------|----------|
| **Focus** | Voice-only | Multi-modal |
| **Dependencies** | ~15 packages | ~35+ packages |
| **Startup Time** | Fast | Moderate |
| **Memory Usage** | Low | High |
| **AI Components** | Essential only | All advanced |
| **GUI** | Beautiful & Simple | Complex multi-screen |

## 🎯 Perfect For:

- **Voice-first users** who prefer speaking over typing
- **Quick interactions** - time, weather, search, chat
- **Minimal setup** - fewer dependencies to install
- **Testing voice features** before using full system
- **Lightweight deployment** on resource-constrained devices

## 🔍 Architecture

```
Voice LEO Structure:
├── Voice_LEO.py           # Main voice-focused application
├── Frontend/GUI.py        # Beautiful animated interface  
├── Backend/
│   ├── RealtimeSearchEngine.py  # Web search
│   └── Chatbot.py              # AI conversation
├── voice_requirements.txt # Minimal dependencies
└── Voice_README.md       # This file
```

## 🎤 Voice Processing Flow

1. **Auto-Listen** → Continuous listening starts automatically
2. **Capture** → Speech-to-text conversion when you speak
3. **Route** → Intelligent command classification
4. **Process** → Search, chat, or built-in response
5. **Respond** → Text-to-speech or text display
6. **Continue** → Returns to listening mode automatically

## 🛠️ Troubleshooting

**Voice not working?**
- Check microphone permissions
- Verify `SpeechRecognition` is installed

**Search not working?**
- Check internet connection
- Verify Groq API key in `.env` file

**GUI not appearing?**
- Install PyQt5: `pip install PyQt5`
- Check system display settings

## 🎉 Success!

When everything works, you'll have:
- A beautiful animated GUI with a robot assistant
- Click-to-talk voice interaction
- Intelligent responses to your voice commands
- Real-time search and chat capabilities
- Minimal resource usage

**Enjoy your streamlined Voice LEO experience!** 🎤✨
