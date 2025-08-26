from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from dotenv import dotenv_values
import os
import mtranslate as mt

# Load environment variables
env_vars = dotenv_values(".env")
InputLanguage = env_vars.get("InputLanguage", "en")

# Create Voice.html
html_content = f'''<!DOCTYPE html>
<html lang="en">
<head><title>Speech Recognition</title></head>
<body>
    <button id="start" onclick="startRecognition()">Start Recognition</button>
    <button id="end" onclick="stopRecognition()">Stop Recognition</button>
    <p id="output"></p>
    <script>
        const output = document.getElementById('output');
        let recognition;

        function startRecognition() {{
            recognition = ('webkitSpeechRecognition' in window)
                ? new webkitSpeechRecognition()
                : new SpeechRecognition();
            recognition.lang = '{InputLanguage}';
            recognition.continuous = true;

            recognition.onresult = function(event) {{
                const transcript = event.results[event.results.length - 1][0].transcript;
                output.textContent += transcript;
            }};

            recognition.onend = function() {{
                recognition.start();
            }};
            recognition.start();
        }}

        function stopRecognition() {{
            recognition.stop();
            output.innerHTML = "";
        }}
    </script>
</body>
</html>'''

# Save Voice.html
os.makedirs("Data", exist_ok=True)
with open("Data/Voice.html", "w") as f:
    f.write(html_content)

# ChromeDriver setup (fixed path)
chrome_options = Options()
chrome_options.add_argument("--use-fake-ui-for-media-stream")
chrome_options.add_argument("--use-fake-device-for-media-stream")
chrome_options.add_argument("--headless=new")  # optional

# 🟢 Fix: Use correct Homebrew path for macOS
service = Service("/opt/homebrew/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=chrome_options)

# Assistant status writing
temp_dir = os.path.join(os.getcwd(), "Frontend", "Files")
os.makedirs(temp_dir, exist_ok=True)

def SetAssistantStatus(status):
    with open(os.path.join(temp_dir, "Status.data"), "w", encoding='utf-8') as f:
        f.write(status)

def QueryModifier(query):
    query = query.strip().lower()
    if query.endswith(('.', '!', '?')):
        query = query[:-1]
    question_words = ["what", "who", "where", "when", "why", "how", "which", "whose", "whom", "can you"]
    if any(query.startswith(word) for word in question_words):
        query += "?"
    else:
        query += "."
    return query.capitalize()

def UniversalTranslator(text):
    translated = mt.translate(text, "en", "auto")
    return translated.capitalize()

def SpeechRecognition():
    html_path = "file://" + os.path.abspath("Data/Voice.html")
    driver.get(html_path)
    driver.find_element(By.ID, "start").click()
    
    while True:
        try:
            output = driver.find_element(By.ID, "output").text
            if output:
                driver.find_element(By.ID, "end").click()
                if "en" in InputLanguage.lower():
                    return QueryModifier(output)
                else:
                    SetAssistantStatus("Translating...")
                    return QueryModifier(UniversalTranslator(output))
        except Exception:
            pass

if __name__ == "__main__":
    print("Listening...")
    while True:
        result = SpeechRecognition()
        print("Recognized:", result)
