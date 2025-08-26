import pygame
import random
import asyncio
import edge_tts
import os
import logging
from dotenv import dotenv_values

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
env_vars = dotenv_values(".env")
AssistantVoice = env_vars.get("AssistantVoice", "en-US-GuyNeural")

# File path for TTS output
SPEECH_FILE_PATH = "Data/speech.mp3"

# Utility to safely run async functions in sync context
def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        return asyncio.get_event_loop().run_until_complete(coro)

# Async function to generate TTS audio
async def TextToAudioFile(text: str) -> None:
    if os.path.exists(SPEECH_FILE_PATH):
        os.remove(SPEECH_FILE_PATH)

    try:
        communicate = edge_tts.Communicate(text, AssistantVoice, pitch='+5Hz', rate='+13%')
        await communicate.save(SPEECH_FILE_PATH)
    except Exception as e:
        logger.error(f"[TextToAudioFile] Edge TTS Error: {e}")
        raise e

# Core TTS playback function
def TTS(text: str, func=lambda r=None: True):
    try:
        run_async(TextToAudioFile(text))

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(SPEECH_FILE_PATH)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            if func() == False:
                break
            pygame.time.Clock().tick(10)

    except Exception as e:
        logger.error(f"[TTS Error] {e}")
        print("[Fallback] " + text)

    finally:
        try:
            func(False)
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            if os.path.exists(SPEECH_FILE_PATH):
                os.remove(SPEECH_FILE_PATH)
        except Exception as e:
            logger.warning(f"[TTS Cleanup Error] {e}")

# TTS wrapper with text shortening logic for long content
def TextToSpeech(text: str, func=lambda r=None: True):
    sentences = text.strip().split(".")
    responses = [
        "The rest of the result has been printed to the chat screen, kindly check it out sir.",
        "The rest of the text is now on the chat screen, sir, please check it.",
        "You can see the rest of the text on the chat screen, sir.",
        "The remaining part of the text is now on the chat screen, sir.",
        "Sir, you'll find more text on the chat screen for you to see.",
        "The rest of the answer is now on the chat screen, sir.",
        "Sir, please look at the chat screen, the rest of the answer is there.",
        "You'll find the complete answer on the chat screen, sir.",
        "The next part of the text is on the chat screen, sir.",
        "Sir, please check the chat screen for more information.",
        "There's more text on the chat screen for you, sir.",
        "Sir, take a look at the chat screen for additional text.",
        "You'll find more to read on the chat screen, sir.",
        "Sir, check the chat screen for the rest of the text.",
        "The chat screen has the rest of the text, sir.",
        "There's more to see on the chat screen, sir, please look.",
        "Sir, the chat screen holds the continuation of the text.",
        "Please review the chat screen for the rest of the text, sir.",
        "Sir, look at the chat screen for the complete answer."
    ]

    # Truncate if too long
    # if len(sentences) > 4 and len(text) >= 250:
    #     summary = ". ".join(sentences[:2]) + ". " + random.choice(responses)
    #     TTS(summary, func)
    # else:
    TTS(text, func)

# CLI testing entry
if __name__ == "__main__":
    while True:
        try:
            user_input = input("Enter the Text: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            TextToSpeech(user_input)
        except KeyboardInterrupt:
            break
