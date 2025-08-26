import platform
from webbrowser import open as webopen
from pywhatkit import search,playonyt
from dotenv import dotenv_values
from bs4 import BeautifulSoup
from rich import print
from groq import Groq
import requests
import webbrowser
import subprocess
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import asyncio
from requests_html import HTMLSession
import urllib
from googlesearch import search


# Load environment variables from .env file
env_vars = dotenv_values(".env")
GroqAPIKey = env_vars.get("GroqAPIKey")

# Define CSS Classes for parsing specific elements in HTML content
classes = ["zCubwf", "hgKElc", "LTKOO sY7ric", "Z0LcW", "gsrt vk_bk FzvWSb", "pclqee", "tw-Data-text tw-text-small tw-ta", "IZ6rdc", "O5uR6d LTKOO", "vlzY6d", "webanswers-webanswers_table__webanswers-table", "dDoNo ikb4Bd gsrt", "sXLaOe", "LWkfKe", "VQF4g", "qv3Wpe", "kno-rdesc", "SPZz6b"]

# Define a user-agent for making web requests
useragent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"

# Initialize Groq client with the API key
client = Groq(api_key=GroqAPIKey)

# Predefined professional response for user interactions
professional_response = [
    "Your satisfaction is my top priority; feel free to reach out if there's anything else I can assist you with; I'm always ready to help.",
    "T'm at your service for any further assistance you may need; don't hesitate to ask."
    ]

# List to store chatbot messages
messages = []

# Define a system message that provides context to the AI chatbot
SystemChatBot = [{"role": "system", "content": f"Hello, I am {env_vars.get('Username', 'User')}, You're a content writer. You have to write contents like letters, codes, applications, essays, notes, songs, poems, etc."}]

# Function to perform a google search
def GoogleSearch(Topic):
    query = Topic.replace(" ", "+")
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return True
# Function to generate content using AI and save it to a file
def Content(Topic):
    
    # nested function to open a file in mac builit in text editor
    def OpenNotepad(File):
        # Open the file in the default text editor (TextEdit on macOS)
        default_text_editor = "TextEdit"
        subprocess.Popen(["open", "-a", default_text_editor, File])
    
    # nested function to generate content using AI chatbot
    def ContentWriterAI(prompt):
        messages.append({"role": "user", "content": f"{prompt}"})
        
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=SystemChatBot + messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )
        
        Answer = ""
        
        # process streamed response chunks
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content
                
        Answer = Answer.replace("</s>", "")
        messages.append({"role": "assistant", "content": Answer})
        return Answer
    
    Topic: str = Topic.replace("Content ", "")
    ContentByAI = ContentWriterAI(Topic)
    
    # Save the generated content to a file
    with open(rf"Data/{Topic.lower().replace(' ','')}.txt", "w", encoding="utf-8") as file:
        file.write(ContentByAI)
        file.close()
        
    OpenNotepad(rf"Data/{Topic.lower().replace(' ','')}.txt")
    return True

# Function to search for a topic on youtube
def YouTubeSearch(Topic):
    Url4Search = f"https://www.youtube.com/results?search_query={Topic}"
    webbrowser.open(Url4Search)
    return True

# Function to play a video on youtube
def PlayYoutube(query):
    playonyt(query)  # Opens and auto-plays top result on YouTube in default browser
    return True

# Function to play songs
# Define your Spotify app credentials and redirect URI
SPOTIFY_CLIENT_ID = "7a5dc65af7ef4fd2b763de19d14153d0"
SPOTIFY_CLIENT_SECRET = "b463f1edbd734cab981b12f8d9e39087"
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback"

# Apps and their macOS bundle names
MUSIC_APPS = {
    "spotify": "Spotify",
    "apple music": "Music",
    "youtube music": None,  # no native app, fallback to browser
}

def PlayMusic(command: str):
    """
    Plays music based on a user command, e.g.:
    - "play perfect on spotify"
    - "play spotify"
    - "play hello by adele on apple music"
    """

    # Basic parsing of command
    command = command.lower()
    found_app = None
    for app_key in MUSIC_APPS.keys():
        if app_key in command:
            found_app = app_key
            break

    if not found_app:
        print("No recognized music app found in command.")
        return False

    song_query = command.replace("play", "").replace(found_app, "").strip()
    app_name = MUSIC_APPS[found_app]

    # Check macOS platform
    if platform.system() != "Darwin":
        print("This function currently only supports macOS.")
        return False

    # If app installed, open and play; else open web player
    def is_app_installed(app):
        ret = subprocess.call(["open", "-Ra", app])
        return ret == 0

    if app_name and is_app_installed(app_name):
        # For Spotify, use Spotify Web API to play song or resume playback
        if found_app == "spotify":
            sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope="user-read-playback-state user-modify-playback-state"
            ))

            devices = sp.devices()
            if not devices['devices']:
                print("No active Spotify device found. Please open Spotify app and start playback once.")
                subprocess.call(["open", "-a", app_name])
                return True

            device_id = devices['devices'][0]['id']

            if song_query:
                results = sp.search(q=song_query, type='track', limit=1)
                tracks = results.get('tracks', {}).get('items', [])
                if tracks:
                    track_uri = tracks[0]['uri']
                    sp.start_playback(device_id=device_id, uris=[track_uri])
                    print(f"Playing '{tracks[0]['name']}' by '{tracks[0]['artists'][0]['name']}' on Spotify.")
                else:
                    print("Song not found on Spotify. Resuming playback.")
                    sp.start_playback(device_id=device_id)
            else:
                sp.start_playback(device_id=device_id)
                print("Resuming current playback on Spotify.")

            subprocess.call(["open", "-a", app_name])
            return True

        # For Apple Music: open app and play (resume or open song URL on web)
        elif found_app == "apple music":
            subprocess.call(["open", "-a", app_name])
            if song_query:
                # Open song search on Apple Music web player
                query = urllib.parse.quote(song_query)
                url = f"https://music.apple.com/us/search?term={query}"
                webbrowser.open(url)
                print(f"Opened Apple Music web search for '{song_query}'.")
            else:
                print("Opened Apple Music app. Please resume playback manually.")
            return True

    # Fallback: open web player for the app
    if found_app == "spotify":
        base_url = "https://open.spotify.com/search/"
    elif found_app == "apple music":
        base_url = "https://music.apple.com/us/search?term="
    elif found_app == "youtube music":
        base_url = "https://music.youtube.com/search?q="
    else:
        print("App not supported.")
        return False

    url = base_url + urllib.parse.quote(song_query) if song_query else base_url
    webbrowser.open(url)
    print(f"Opened web player for {found_app} with query '{song_query}'")
    return True 

# Function to open an application or a relevant webpage
def OpenApp(app):
    import platform
    import subprocess

    if platform.system() == "Darwin":
        ret = subprocess.call(["open", "-a", app])
        if ret == 0:
            return True

        print(f"[App not found: {app}] Opening first Google search result in browser...")

        try:
            # Just iterate over search results without num or stop args
            results = search(app)
            first_url = next(results, None)
            if first_url:
                webbrowser.open(first_url)
                return True
            else:
                print("[Error]: No valid Google search result found.")
                return False
        except Exception as e:
            print(f"[Error in Google Search]: {e}")
            return False

    else:
        raise NotImplementedError("Only macOS is supported.")

#Function to close an application
def CloseApp(app):
    
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.call(["osascript", "-e", f'quit app "{app}"'])
        else:
            raise NotImplementedError("This function is only implemented for macOS.")
        return True
    except Exception as e:
        print(f"[Error closing app]: {e}")
        return False

# Function to execute system-level commands
def System(command):
    def run_applescript(script):
        subprocess.run(["osascript", "-e", script])

    if command == "mute":
        run_applescript("set volume output muted true")
    elif command == "unmute":
        run_applescript("set volume output muted false")
    elif command == "volume up":
        # Increase volume by 10%
        run_applescript("set volume output volume ((output volume of (get volume settings)) + 10) --100% max")
    elif command == "volume down":
        # Decrease volume by 10%
        run_applescript("set volume output volume ((output volume of (get volume settings)) - 10) --0% min")
    else:
        print(f"Unknown system command: {command}")
        return False

    return True

# Asynchronous function to translate and execute user commands
async def TranslateAndExecute(commands: list[str]):
    funcs = []

    for command in commands:
        if command.startswith("open "):
            fun = asyncio.to_thread(OpenApp, command.removeprefix("open "))
            funcs.append(fun)

        elif command.startswith("close "):
            fun = asyncio.to_thread(CloseApp, command.removeprefix("close "))
            funcs.append(fun)

        elif command.startswith("play ") and command.endswith(" on youtube"):
            fun = asyncio.to_thread(PlayYoutube, command.removeprefix("play "))
            funcs.append(fun)

        elif command.startswith("play "):
            # Handle playing music on Spotify, Apple Music, YouTube Music, etc.
            fun = asyncio.to_thread(PlayMusic, command)
            funcs.append(fun)

        elif command.startswith("content "):
            fun = asyncio.to_thread(Content, command.removeprefix("content "))
            funcs.append(fun)

        elif command.startswith("google search "):
            fun = asyncio.to_thread(GoogleSearch, command.removeprefix("google search "))
            funcs.append(fun)

        elif command.startswith("youtube search "):
            fun = asyncio.to_thread(YouTubeSearch, command.removeprefix("youtube search "))
            funcs.append(fun)

        elif command.startswith("system "):
            fun = asyncio.to_thread(System, command.removeprefix("system "))
            funcs.append(fun)

        else:
            print(f"Unknown command: {command}")

    results = await asyncio.gather(*funcs)

    for result in results:
        yield result
            
# Aysnchronous function to automate the execution of commands
async def Automation(commands: list[str]):
    
    async for result in TranslateAndExecute(commands):
        pass
    
    return True


if __name__ == "__main__":
    # Example usage
    commands = [
        "open Safari",
        "play perfect on spotify",
        "content write a letter to my friend",
        "google search Python programming",
        "youtube search funny cat videos"
    ]
    
    asyncio.run(Automation(commands))
